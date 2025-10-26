# src/training/orchestrator.py
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import confusion_matrix, f1_score
except ImportError:  # pragma: no cover
    def f1_score(y_true, y_pred, average="macro"):
        return 0.0

    def confusion_matrix(y_true, y_pred):
        return [[0]]

from .utils.probe import fit_probe_and_log
from .utils.data import build_sl_loaders, build_ssl_loader, device_from_env, class_labels_from_cfg
from .utils.io import (
    append_row_csv,
    copy_yaml_config,
    dump_json,
    make_exp_id,
    make_run_dirs,
    prefixed,
    save_env_info,
    save_state_dict,
    write_run_readme,
)
from .utils.trainers import SSLTrainer, SLTrainer, safe_state_dict
from .utils.viz import plot_confusion, render_all_sl, render_all_ssl, write_derived_csv

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
        self.exp_id = make_exp_id(cfg["experiment"]["outputs_root"])
        self.run_dirs = make_run_dirs(cfg["experiment"]["outputs_root"], self.exp_id, cfg["experiment"]["name"], self.model_key)
        config_path = self.cfg.get("_runtime", {}).get("config_path") or os.environ.get("EXPERIMENT_CONFIG_PATH")
        copy_yaml_config(config_path, self.run_dirs["configuration"])
        save_env_info(self.run_dirs["configuration"], self.cfg["experiment"].get("seed", 1337))
        dump_json(self.run_dirs["configuration"] / "resolved_config.json", self.cfg)
        self.override_transforms: Optional[Any] = None

    def fit(self) -> Dict[str, float]:
        if self.mode == "ssl":
            metrics = self._fit_ssl()
        elif self.mode == "sl":
            metrics = self._fit_sl()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        self._finalize_run(metrics)
        return metrics

    def _finalize_run(self, metrics: Dict[str, float]) -> None:
        write_run_readme(self.run_dirs, self.model_key, self.mode, self.cfg, metrics, self.exp_id)

    # ------------------------------------------------------------------ helpers
    def _build_optimizer(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        conf = self.cfg["train"]["optim"]
        name = conf.get("name", "adamw").lower()
        lr = conf["lr"]
        weight_decay = conf.get("weight_decay", 5e-2)
        if name == "adamw":
            betas = tuple(conf.get("betas", (0.9, 0.999)))
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        if name == "sgd":
            momentum = conf.get("momentum", 0.9)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer '{name}'.")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        sched_cfg = (self.cfg["train"].get("scheduler") or {})
        name = sched_cfg.get("name", "").lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return None

    # ------------------------------------------------------------------ SSL path
    def _fit_ssl(self) -> Dict[str, float]:
        model = _with_context("build_ssl_model", _ssl_model_factory, self.cfg)
        loader = _with_context("build_ssl_loader", build_ssl_loader, self.cfg["data"], self.cfg["model"], "train")
        print(f"[RUN][{self.model_key}] device={self.device.type}")

        ssl_cfg = self.cfg["train"]["ssl"]
        optimizer = self._build_optimizer(model.parameters())
        scheduler = self._build_scheduler(optimizer, int(ssl_cfg["epochs"] * ssl_cfg.get("steps_per_epoch", len(loader))))
        trainer = SSLTrainer(
            model,
            optimizer,
            scheduler=scheduler,
            device=self.device,
            log_every_steps=_log_every_steps(self.cfg),
            log_tag=self.model_key,
        )

        steps_per_epoch = ssl_cfg.get("steps_per_epoch")
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(loader)
            except TypeError:
                raise ValueError("train.ssl.steps_per_epoch must be set when the loader has no __len__.")
        steps_per_epoch = max(1, int(steps_per_epoch))
        epochs = int(ssl_cfg["epochs"])

        csv_path = prefixed(self.run_dirs["metrics"], self.model_key, "ssl_timeseries", "csv")
        best_loss = float("inf")
        best_epoch = -1
        best_state = None
        best_path = prefixed(self.run_dirs["artifacts"], self.model_key, "ssl_best", "pt")
        global_step_offset = 0

        def _epoch_mode() -> Callable:
            name = self.cfg["model"]["ssl"]["name"].lower()
            if name in ("moco_v3", "ibot"):
                return trainer.train_epoch_two_views
            if name == "dino_v3":
                return trainer.train_epoch_multicrop
            if name == "i_jepa":
                return trainer.train_epoch_single_image
            raise ValueError(f"Unsupported SSL model '{name}'.")

        run_epoch = _epoch_mode()

        for epoch in range(epochs):
            def _log_step(global_step: int, stats: Dict[str, float], epoch_idx: int = epoch) -> None:
                row = {"epoch": epoch_idx, "step": global_step, "lr": optimizer.param_groups[0].get("lr", 0.0)}
                row.update(stats)
                append_row_csv(csv_path, row)

            epoch_stats = run_epoch(loader, steps_per_epoch, start_step=global_step_offset, step_callback=_log_step)
            global_step_offset += steps_per_epoch
            loss_epoch = float(epoch_stats.get("loss_total", float("inf")))
            if loss_epoch < best_loss:
                best_loss = loss_epoch
                best_epoch = epoch
                best_state = safe_state_dict(model)
                best_path = save_state_dict(best_state, self.run_dirs["artifacts"], self.model_key, "ssl_best")

            if hasattr(model, "on_epoch_end"):
                model.on_epoch_end(epoch)

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
            model.to(self.device)

        derived_csv = prefixed(self.run_dirs["plots"], self.model_key, "ssl_timeseries_derived", "csv")
        write_derived_csv(csv_path, derived_csv)
        render_all_ssl(derived_csv, self.run_dirs["figures"], self.model_key)

        metrics_path = prefixed(self.run_dirs["metrics"], self.model_key, "ssl_final_metrics", "json")
        dump_json(metrics_path, {"best_epoch": int(best_epoch), "ssl_loss": float(best_loss)})

        ssl_summary = {
            "best_epoch": int(best_epoch),
            "ssl_loss": float(best_loss),
            "ssl_best_ckpt_path": str(best_path),
            "ssl_metrics_path": str(metrics_path),
        }

        probe_cfg = (self.cfg.get("train", {}).get("ssl", {}) or {}).get("probe", {}) or {}
        run_probe = probe_cfg.get("enabled", True)
        if run_probe:
            run_probe = probe_cfg.get("do_linear_probe", True) or probe_cfg.get("do_knn", False)

        if run_probe:
            train_loader, val_loader = _with_context(
                "build_sl_loaders",
                build_sl_loaders,
                self.cfg,
                override_transforms=self.override_transforms,
            )
            loaders = {"train": train_loader, "val": val_loader}
            print(f"[RUN][{self.model_key}→PROBE] device={self.device.type}")
            backbone = model.stu if hasattr(model, "stu") else model
            probe_metrics = _with_context(
                "ssl_probe",
                fit_probe_and_log,
                backbone,
                loaders,
                self.run_dirs,
                self.model_key,
                self.cfg,
                self.device,
            )
            return {**ssl_summary, **probe_metrics}

        return ssl_summary

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
        best_path = prefixed(self.run_dirs["artifacts"], self.model_key, "sl_best_classifier", "pt")
        epochs = int(self.cfg["train"]["sl"]["epochs"])

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
                },
            )

            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                best_state = safe_state_dict(model)
                best_path = save_state_dict(best_state, self.run_dirs["artifacts"], self.model_key, "sl_best_classifier")

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
            model.to(self.device)

        dcsv = prefixed(self.run_dirs["plots"], self.model_key, "sl_timeseries_derived", "csv")
        write_derived_csv(csv_path, dcsv)
        render_all_sl(dcsv, self.run_dirs["figures"], self.model_key)

        final_metrics = self._evaluate_sl(model, loaders["val"], criterion)
        metrics_path = prefixed(self.run_dirs["metrics"], self.model_key, "sl_final_metrics", "json")
        dump_json(
            metrics_path,
            {
                "best_epoch": int(best_epoch),
                "val_acc": float(final_metrics["val_acc"]),
                "val_f1_macro": float(final_metrics["val_f1_macro"]),
                "val_loss": float(final_metrics["val_loss"]),
            },
        )

        return {
            "best_epoch": int(best_epoch),
            "val_acc": float(final_metrics["val_acc"]),
            "val_loss": float(final_metrics["val_loss"]),
            "sl_best_ckpt_path": str(best_path),
            "sl_metrics_path": str(metrics_path),
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
        plot_confusion(cm, labels, prefixed(self.run_dirs["figures"], self.model_key, "sl_confusion_val", "png"))
        return {
            "val_loss": avg_loss,
            "val_acc": avg_acc,
            "val_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
