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
