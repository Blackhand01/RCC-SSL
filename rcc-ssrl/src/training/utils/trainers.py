# utils/trainers.py
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

__all__ = [
    "l2n",
    "cosine_logits",
    "ema_update",
    "copy_weights_and_freeze",
    "move_to",
    "ResNetBackbone",
    "mlp_head",
    "predictor_head",
    "safe_state_dict",
    "SSLBaseModel",
    "SLBaseModel",
    "SSLTrainer",
    "SLTrainer",
]

# -----------------------------------------------------------------------------
# Primitive ops
# -----------------------------------------------------------------------------
def l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalizza sull'ultima dimensione evitando divisioni per zero."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _safe_tau(tau: float) -> float:
    return max(tau, 1e-8)


def cosine_logits(q: torch.Tensor, k: torch.Tensor, tau: float) -> torch.Tensor:
    """Logit di similarità coseno con temperatura."""
    return (l2n(q) @ l2n(k).t()) / _safe_tau(tau)


def move_to(obj: Any, device: torch.device) -> Any:
    """Sposta ricorsivamente tensori su device, mantenendo struttura."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [move_to(v, device) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
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

# -----------------------------------------------------------------------------
# Backbones & heads
# -----------------------------------------------------------------------------
def _get_resnet_factory(name: str):
    factories = {"resnet34": models.resnet34, "resnet50": models.resnet50}
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
        self.layer1, self.layer2, self.layer3, self.layer4 = model.layer1, model.layer2, model.layer3, model.layer4
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


def _bn1d(dim: int, affine: bool = True) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(dim, affine=affine)


def _relu() -> nn.ReLU:
    return nn.ReLU(inplace=True)


def _linear(in_f: int, out_f: int, bias: bool = False) -> nn.Linear:
    return nn.Linear(in_f, out_f, bias=bias)


def mlp_head(in_dim: int, hidden: int, out_dim: int, bn_last_affine: bool = False) -> nn.Sequential:
    """MLP 3-layer con BN e ReLU; ultima BN opzionale affine."""
    return nn.Sequential(
        _linear(in_dim, hidden, bias=False), _bn1d(hidden), _relu(),
        _linear(hidden, hidden, bias=False), _bn1d(hidden), _relu(),
        _linear(hidden, out_dim, bias=False), _bn1d(out_dim, affine=bn_last_affine),
    )


def predictor_head(dim: int, hidden: int = 4096) -> nn.Sequential:
    """MLP predittore stile BYOL/MoCoV3."""
    return nn.Sequential(
        _linear(dim, hidden, bias=False), _bn1d(hidden), _relu(),
        _linear(hidden, dim),
    )

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
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema_m = float(ema_m)
        self.device = device or (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
        self.log_every = int(log_every_steps)
        self.log_tag = log_tag or model.__class__.__name__
        self.model.to(self.device)

    # ---- internals ----------------------------------------------------------
    def _run_step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, float]:
        batch = move_to(batch, self.device)
        out = self.model.training_step(batch, global_step)
        loss = out["loss_total"]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        comp = {k: float(v) for k, v in out.get("loss_components", {}).items()}
        comp["loss_total"] = float(loss.detach())
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

            if self.log_every and (s + 1) % self.log_every == 0:
                now = time.time()
                dt = now - last
                last = now
                eta = _eta_secs(t0, s + 1, steps)
                ema_msg = " ".join(f"{k}={metrics.ema[k]:.4f}" for k in sorted(metrics.ema))
                print(f"[{self.log_tag}][step {s+1}/{steps}] ETA={int(eta//60):02d}:{int(eta%60):02d} dt/step={dt:.2f}s {ema_msg}", flush=True)

        avg = metrics.averaged(steps)
        avg["steps"] = steps
        avg.update({f"{k}_ema": v for k, v in metrics.ema.items()})
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
# Nota: l’uso di autocast/GradScaler segue le linee guida AMP ufficiali.  # docs
# https://pytorch.org/docs/stable/notes/amp_examples.html
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
            import torch.amp as _amp  # PyTorch ≥ 2
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
            return batch["inputs"].to(device, non_blocking=True), batch["targets"].to(device, non_blocking=True)
        # compat: alcuni loader forniscono "images"/"label"
        return batch["images"][0].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)

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
                    print(f"[{self.log_tag}][{'train' if train else 'val'}] "
                          f"[{min(idx, total_batches)}/{total_batches}] ETA={eta} "
                          f"loss={loss:.4f} acc={acc:.4f} dt/step={time.time() - t0:.2f}s", flush=True)
                else:
                    print(f"[{self.log_tag}][{'train' if train else 'val'}] "
                          f"[step {idx}] loss={loss:.4f} acc={acc:.4f} dt/step={time.time() - t0:.2f}s", flush=True)

        if train and self.scheduler is not None:
            self.scheduler.step()

        denom = max(1, total_samples)
        return {"loss": total_loss / denom, "acc": total_acc / denom}
