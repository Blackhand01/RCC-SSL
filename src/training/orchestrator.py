
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

# ---- models (reusing yours) ----
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
        # Note: in SSL we use 'steps' as unit; in SL 'epochs'
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
        
        if hasattr(model, "set_total_steps"):
            model.set_total_steps(total_steps)
        elif hasattr(model, "total_steps"):
            model.total_steps = total_steps

        # Cosine on 'units' of the program (for SSL=steps, for SL=epochs)
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

        # ------------------------------------------------------------------ time budget (optional)
        # Read max runtime from env (in hours). Example: MAX_RUNTIME_HOURS=23.5
        max_runtime_hours_env = os.environ.get("MAX_RUNTIME_HOURS", "").strip()
        max_runtime_s: float | None = None
        if max_runtime_hours_env:
            try:
                h = float(max_runtime_hours_env)
                if h > 0:
                    max_runtime_s = h * 3600.0
                    if os.environ.get("RANK", "0") == "0":
                        print(f"[time_budget] MAX_RUNTIME_HOURS={h:.2f}h (≈{max_runtime_s/3600:.2f}h)")
            except ValueError:
                if os.environ.get("RANK", "0") == "0":
                    print(f"[time_budget] WARNING: invalid MAX_RUNTIME_HOURS='{max_runtime_hours_env}' (ignored)")

        
        from pathlib import Path as _P
        csv_stem = _P((self.cfg.get("logging", {}) or {}).get("metrics_csv_name", "ssl_timeseries.csv")).stem
        csv_path = prefixed(self.run_dirs["metrics"], self.model_key, csv_stem, "csv")
        #---blocco nuovo test--------
        # best_loss = float("inf")
        # best_epoch = -1
        # best_state = None
        #ssl_cfg = self.cfg["train"]["ssl"]
        ckpt_cfg = (ssl_cfg.get("checkpoint") or {})
        ckpt_metric = ckpt_cfg.get("metric", "ssl_loss")  # default = current behavior
        warmup_epochs = int(ckpt_cfg.get("warmup_epochs", 0))
        eval_every = int(ckpt_cfg.get("eval_every_epochs", 1))

        best_ssl_loss = float("inf")
        best_probe_acc = -1.0
        best_epoch = -1
        best_state = None
        
        probe_loaders = None
        probe_cfg = (ssl_cfg.get("probe") or {})
        checkpoint_probe_epochs = int(
            ckpt_cfg.get("probe_epochs", probe_cfg.get("epochs", 5))
        )
        if ckpt_metric == "probe_val_acc":
            # reuse the same logic as SL loaders
            train_loader, val_loader = _with_context(
                "build_sl_loaders_for_probe",
                build_sl_loaders,
                self.cfg,
                override_transforms=self.override_transforms,
            )
            probe_loaders = {"train": train_loader, "val": val_loader}

        #----fine blocco nuovo test----
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
            #---blocco nuovo test--------
            # if loss_epoch < best_loss:
            #     best_loss = loss_epoch
            #     best_epoch = epoch
            #     best_state = safe_state_dict(model)
            
            if ckpt_metric == "ssl_loss":
                if loss_epoch < best_ssl_loss:
                    best_ssl_loss = loss_epoch
                    best_epoch = epoch
                    best_state = safe_state_dict(model)

            elif ckpt_metric == "ssl_loss_after_warmup":
                if epoch >= warmup_epochs and loss_epoch < best_ssl_loss:
                    best_ssl_loss = loss_epoch
                    best_epoch = epoch
                    best_state = safe_state_dict(model)

            elif ckpt_metric == "final_epoch":
                # sovrascrivi sempre: best = ultimo
                best_ssl_loss = loss_epoch
                best_epoch = epoch
                best_state = safe_state_dict(model)

            elif ckpt_metric == "probe_val_acc":
                # valuta solo dopo il warmup, ogni eval_every epoche
                if (epoch >= warmup_epochs) and ((epoch - warmup_epochs) % max(1, eval_every) == 0):
                    if probe_loaders is None:
                        raise RuntimeError("probe_val_acc selected but probe_loaders is None")

                    # Current backbone (for I-JEPA it is the student)
                    backbone_module = model.stu if hasattr(model, "stu") else model

                    # Extract features on train / val
                    tag = f"{self.model_key}_ep{epoch:03d}"
                    feature_paths = save_features(
                        backbone_module,
                        probe_loaders,
                        self.device,
                        self.run_dirs["checkpoints"],
                        tag,
                    )
                    Xtr = np.load(feature_paths["train_X"], allow_pickle=False)
                    ytr = np.load(feature_paths["train_y"], allow_pickle=False)
                    Xva = np.load(feature_paths["val_X"], allow_pickle=False)
                    yva = np.load(feature_paths["val_y"], allow_pickle=False)

                    # Train light linear probe
                    lin_metrics, lin_ckpt = train_linear_probe_torch(
                        Xtr,
                        ytr,
                        Xva,
                        yva,
                        n_epochs=checkpoint_probe_epochs,
                        lr=float(probe_cfg.get("lr", 0.01)),
                        wd=float(probe_cfg.get("weight_decay", 0.0)),
                        batch_size=int(probe_cfg.get("batch_size", 256)),
                        out_dirs=self.run_dirs,
                        model_key=tag,
                    )
                    cur_acc = float(lin_metrics.get("val_acc", float("nan")))
                    print(
                        f"[{self.model_key}][epoch {epoch+1}/{epochs}] "
                        f"probe_val_acc={cur_acc:.4f} (best={best_probe_acc:.4f})",
                        flush=True,
                    )

                    # Update best if improves
                    if cur_acc > best_probe_acc:
                        best_probe_acc = cur_acc
                        best_epoch = epoch
                        best_state = safe_state_dict(model)
                        # Also track the ssl_loss related to that epoch (optional)
                        best_ssl_loss = loss_epoch

            else:
                raise ValueError(f"Unsupported checkpoint.metric='{ckpt_metric}'")
            
            #---fine blocco nuovo test----
            if hasattr(model, "on_epoch_end"):
                model.on_epoch_end(epoch)
            
            if max_runtime_s is not None:
                elapsed = time.time() - t0_run
                if elapsed >= max_runtime_s:
                    if os.environ.get("RANK", "0") == "0":
                        h = elapsed / 3600.0
                        print(
                            f"[time_budget][{self.model_key}] "
                            f"Reached time budget ({h:.2f}h >= {max_runtime_s/3600.0:.2f}h). "
                            f"Stopping after epoch {epoch + 1}/{epochs}.",
                            flush=True,
                        )
                    break
        
        best_loss = best_ssl_loss
        
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
            # Estimate ETA end of training (linear on times per epoch)
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
