#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description="Re-train linear probe from saved SSL features (standalone, no torch.optim).")
    p.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Root della run SSL (cartella che contiene checkpoints/, metrics/, plots/).",
    )
    p.add_argument(
        "--model-key",
        type=str,
        default="i_jepa",
        help="Prefix usato per i file di features (default: i_jepa).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Numero di epoche per il linear probe.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate del linear probe.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay L2 del linear probe.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size del linear probe.",
    )
    return p.parse_args()


def train_linear_probe_standalone(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    n_epochs: int,
    lr: float,
    wd: float,
    batch_size: int,
    ckpt_dir: Path,
    metrics_dir: Path,
    model_key: str,
):
    """
    Versione minimale di train_linear_probe_torch:
    - nessun uso di torch.optim (evita torch._dynamo / onnx / transformers).
    - SGD manuale sui pesi del linear head.
    """
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    yva = np.asarray(yva, dtype=np.int64)

    if Xtr.size == 0 or Xva.size == 0:
        raise RuntimeError("Feature arrays vuote, impossibile allenare il probe.")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    in_dim = Xtr.shape[1]
    classes = np.unique(np.concatenate([ytr, yva])) if ytr.size or yva.size else np.arange(in_dim)
    num_classes = int(len(classes)) if len(classes) > 0 else 1

    head = nn.Linear(in_dim, num_classes).to(device)

    # Pesi di classe = inv. frequenza (normalizzati)
    _, counts = np.unique(ytr, return_counts=True)
    w = counts.max() / np.maximum(counts, 1)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{model_key}_ssl_linear_best.pt"
    csv_path = metrics_dir / f"{model_key}_ssl_linear_timeseries.csv"

    def _iter_batches(X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        if N == 0:
            return
        for start in range(0, N, batch_size):
            stop = min(N, start + batch_size)
            xb = torch.from_numpy(X[start:stop]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y[start:stop]).to(device=device, dtype=torch.long)
            yield xb, yb

    best_acc = -1.0

    with csv_path.open("w") as fcsv:
        fcsv.write("epoch,train_loss,val_loss,val_acc,lr\n")

        for epoch in range(max(1, n_epochs)):
            # -------------------- TRAIN --------------------
            head.train()
            total_loss = 0.0
            total_count = 0

            for xb, yb in _iter_batches(Xtr, ytr):
                head.zero_grad(set_to_none=True)
                logits = head(xb)
                loss = criterion(logits, yb)

                # L2 weight decay manuale
                if wd > 0.0:
                    l2 = 0.0
                    for p in head.parameters():
                        if p.requires_grad:
                            l2 = l2 + p.pow(2).sum()
                    loss = loss + 0.5 * wd * l2

                loss.backward()

                # SGD manuale: p = p - lr * grad
                with torch.no_grad():
                    for p in head.parameters():
                        if p.grad is not None:
                            p.add_(p.grad, alpha=-lr)

                total_loss += float(loss.detach()) * yb.size(0)
                total_count += yb.size(0)

            train_loss = total_loss / max(1, total_count)

            # -------------------- VALIDATION --------------------
            head.eval()
            val_loss = 0.0
            val_count = 0
            correct = 0
            with torch.no_grad():
                for xb, yb in _iter_batches(Xva, yva):
                    logits = head(xb)
                    loss = criterion(logits, yb)
                    if wd > 0.0:
                        l2 = 0.0
                        for p in head.parameters():
                            if p.requires_grad:
                                l2 = l2 + p.pow(2).sum()
                        loss = loss + 0.5 * wd * l2

                    val_loss += float(loss.detach()) * yb.size(0)
                    val_count += yb.size(0)
                    pred = logits.argmax(dim=1)
                    correct += int((pred == yb).sum())

            val_loss_avg = val_loss / max(1, val_count)
            val_acc = correct / max(1, val_count)

            fcsv.write(
                f"{epoch},{train_loss:.6f},{val_loss_avg:.6f},{val_acc:.6f},{lr:.6f}\n"
            )
            fcsv.flush()

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {"state_dict": head.state_dict(), "in_dim": in_dim, "num_classes": num_classes},
                    best_path,
                )

            print(
                f"[probe][epoch {epoch+1}/{n_epochs}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f}",
                flush=True,
            )

    return {"val_acc": best_acc}, str(best_path)


def main():
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    ckpt_dir = run_root / "checkpoints"
    metrics_dir = run_root / "metrics"
    plots_dir = run_root / "plots"  # non usato, ma manteniamo la struttura

    # Carica features (quelle finali)
    feat_dir = ckpt_dir / "features"
    Xtr = np.load(feat_dir / f"{args.model_key}_train_X.npy", allow_pickle=False)
    ytr = np.load(feat_dir / f"{args.model_key}_train_y.npy", allow_pickle=False)
    Xva = np.load(feat_dir / f"{args.model_key}_val_X.npy", allow_pickle=False)
    yva = np.load(feat_dir / f"{args.model_key}_val_y.npy", allow_pickle=False)

    print(
        f"[probe] Re-training linear probe for '{args.model_key}' "
        f"for {args.epochs} epochs on saved features..."
    )
    metrics, ckpt_path = train_linear_probe_standalone(
        Xtr,
        ytr,
        Xva,
        yva,
        n_epochs=args.epochs,
        lr=args.lr,
        wd=args.weight_decay,
        batch_size=args.batch_size,
        ckpt_dir=ckpt_dir,
        metrics_dir=metrics_dir,
        model_key=f"{args.model_key}_probe{args.epochs}",
    )

    print(f"[probe] Done. Best val_acc={metrics.get('val_acc', float('nan')):.4f}")
    print(f"[probe] Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    raise SystemExit(main())
