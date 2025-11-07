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
