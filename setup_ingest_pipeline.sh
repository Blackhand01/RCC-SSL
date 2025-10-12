#!/usr/bin/env bash
set -euo pipefail

# Radici fisse
HOME_ROOT="/home/mla_group_01/rcc-ssrl"
SCRATCH_ROOT="/beegfs-scratch/mla_group_01/rcc-ssrl"

echo "[INFO] Setup directory tree…"
mkdir -p "$HOME_ROOT/src/ingest" \
         "$HOME_ROOT/slurm/ingest" \
         "$HOME_ROOT/configs" \
         "$SCRATCH_ROOT/data/processed" \
         "$SCRATCH_ROOT/logs"

# ──────────────────────────────────────────────────────────────────────────────
# config_ingest.py
# ──────────────────────────────────────────────────────────────────────────────
cat > "$HOME_ROOT/src/ingest/config_ingest.py" <<'PY'
from pathlib import Path

HOME_ROOT   = Path("/home/mla_group_01/rcc-ssrl")
SCRATCH_ROOT= Path("/beegfs-scratch/mla_group_01/rcc-ssrl")
DATA_ROOT   = SCRATCH_ROOT / "data"
META_ROOT   = HOME_ROOT / "configs"
OUT_MANIFEST= SCRATCH_ROOT / "data" / "processed" / "rcc_manifest.parquet"
OUT_DATACARD= SCRATCH_ROOT / "data" / "processed" / "rcc_datacard.yaml"
LOGS_DIR    = SCRATCH_ROOT / "logs"

CC_MAP      = META_ROOT / "ccRCC_mapping.json"
PR_MAP      = META_ROOT / "pRCC_mapping.json"
CH_MAP      = META_ROOT / "CHROMO_patient_mapping.json"
ON_MAP      = META_ROOT / "ONCO_patient_mapping.json"

SCAN_ROOTS  = [ DATA_ROOT / "raw" ]

FILE_EXT    = {".svs", ".scn", ".tif", ".tiff"}
HASH_ALGO   = "none"   # "none"|"md5"|"sha1"|"xxh3"
MAX_WORKERS = 8
DRYRUN      = False

WSI_EXT = {".svs", ".scn"}
ROI_EXT = {".tif", ".tiff", ".svs"}
XML_EXT = {".xml"}
PY

# ──────────────────────────────────────────────────────────────────────────────
# step_01_scan.py
# ──────────────────────────────────────────────────────────────────────────────
cat > "$HOME_ROOT/src/ingest/step_01_scan.py" <<'PY'
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List
from .config_ingest import SCAN_ROOTS, FILE_EXT

def iter_files(roots: Iterable[Path], exts: set[str]) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts | {".xml"}:
                out.append(p.resolve())
    return out

def build_lookup(files: List[Path]) -> Dict[str, List[Path]]:
    lk: Dict[str, List[Path]] = {}
    for p in files:
        lk.setdefault(p.name, []).append(p)
    return lk

def main() -> Dict[str, List[Path]]:
    files = iter_files(SCAN_ROOTS, FILE_EXT | {".xml"})
    return build_lookup(files)

if __name__ == "__main__":
    lk = main()
    print(f"[scan] indexed files: {sum(len(v) for v in lk.values())}")
PY

# ──────────────────────────────────────────────────────────────────────────────
# step_02_load_maps.py
# ──────────────────────────────────────────────────────────────────────────────
cat > "$HOME_ROOT/src/ingest/step_02_load_maps.py" <<'PY'
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
from .config_ingest import CC_MAP, PR_MAP, CH_MAP, ON_MAP

MAP_FILES = [CC_MAP, PR_MAP, CH_MAP, ON_MAP]

def infer_class(stem: str) -> str:
    s = stem.lower()
    if "cc" in s and "prcc" not in s: return "ccRCC"
    if "prcc" in s: return "pRCC"
    if "chromo" in s or "chrom" in s: return "CHROMO"
    if "onco" in s: return "ONCO"
    raise ValueError(f"cannot infer class from: {stem}")

def main() -> Dict[str, Dict[str, List[str]]]:
    buckets: Dict[str, Dict[str, List[str]]] = {}
    for mp in MAP_FILES:
        data = json.loads(Path(mp).read_text())
        cl = infer_class(Path(mp).stem)
        buckets[cl] = {str(k): list(v) for k, v in data.items()}
    return buckets

if __name__ == "__main__":
    d = main()
    for k in sorted(d):
        print(k, len(d[k]))
PY

# ──────────────────────────────────────────────────────────────────────────────
# step_03_build_manifest.py
# ──────────────────────────────────────────────────────────────────────────────
cat > "$HOME_ROOT/src/ingest/step_03_build_manifest.py" <<'PY'
from __future__ import annotations
import os, hashlib, concurrent.futures as cf
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from .config_ingest import (WSI_EXT, ROI_EXT, XML_EXT, HASH_ALGO, MAX_WORKERS, OUT_MANIFEST)
from .step_02_load_maps import main as load_maps

try:
    import xxhash  # type: ignore
except Exception:
    xxhash = None

def find_xml_for_slide(slide_path: Path) -> Optional[Path]:
    if not slide_path: return None
    c = slide_path.with_suffix(".xml")
    if c.exists(): return c
    for sib in slide_path.parent.glob("*.xml"):
        if slide_path.stem.split(".")[0] in sib.stem:
            return sib
    return None

def compute_hash(path: Path, algo: str) -> Optional[str]:
    if algo == "none": return None
    if algo == "xxh3":
        if xxhash is None: raise RuntimeError("xxhash not installed")
        h = xxhash.xxh3_128()
        with path.open("rb") as f:
            for ch in iter(lambda: f.read(8*1024*1024), b""):
                h.update(ch)
        return h.hexdigest()
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for ch in iter(lambda: f.read(8*1024*1024), b""):
            h.update(ch)
    return h.hexdigest()

def main(lookup: Dict[str, List[Path]]) -> pd.DataFrame:
    maps = load_maps()
    rows: List[dict] = []

    def add_wsi(cl: str, pid: str, slide_name: str):
        wsi_path = lookup.get(Path(slide_name).name, [None])[0]
        xml_path = find_xml_for_slide(wsi_path) if wsi_path else None
        rows.append({
            "patient_id": pid, "slide_id": slide_name, "class_label": cl, "source": "WSI",
            "wsi_path": str(wsi_path) if wsi_path else None,
            "roi_xml_path": str(xml_path) if xml_path else None,
            "roi_files": None, "has_xml": bool(xml_path), "has_wsi": bool(wsi_path),
            "has_roi": False, "filesize_bytes": os.path.getsize(wsi_path) if wsi_path else None,
            "filehash": None, "notes": None,
        })

    def add_roi(cl: str, pid: str, roi_names: List[str]):
        roi_paths: List[str] = []
        for rn in roi_names:
            cand = lookup.get(Path(rn).name)
            if cand: roi_paths.append(str(cand[0]))
        rows.append({
            "patient_id": pid, "slide_id": None, "class_label": cl, "source": "ROI",
            "wsi_path": None, "roi_xml_path": None, "roi_files": roi_paths if roi_paths else None,
            "has_xml": False, "has_wsi": False, "has_roi": bool(roi_paths),
            "filesize_bytes": None, "filehash": None, "notes": None,
        })

    for cl, bucket in maps.items():
        for pid, items in bucket.items():
            wsi_items = [it for it in items if Path(it).suffix.lower() in WSI_EXT]
            roi_items = [it for it in items if Path(it).suffix.lower() in ROI_EXT]
            for slide in wsi_items: add_wsi(cl, pid, Path(slide).name)
            if roi_items: add_roi(cl, pid, roi_items)

    df = pd.DataFrame(rows)

    if HASH_ALGO != "none":
        def _hash(idx_row):
            i, r = idx_row
            target = r.get("wsi_path") or (r.get("roi_files") or [None])[0]
            if not target: return i, None
            try: return i, compute_hash(Path(target), HASH_ALGO)
            except Exception: return i, None
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, h in ex.map(_hash, df.iterrows()):
                df.at[i, "filehash"] = h

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_MANIFEST, index=False)
    return df

if __name__ == "__main__":
    from .step_01_scan import main as scan
    lk = scan()
    df = main(lk)
    print(df.head())
PY

# ──────────────────────────────────────────────────────────────────────────────
# step_04_validate.py
# ──────────────────────────────────────────────────────────────────────────────
cat > "$HOME_ROOT/src/ingest/step_04_validate.py" <<'PY'
from __future__ import annotations
import sys, pandas as pd
from .config_ingest import OUT_MANIFEST

REQUIRED_COLS = [
    "patient_id","class_label","source",
    "wsi_path","roi_xml_path","roi_files",
    "has_xml","has_wsi","has_roi","notes"
]

def main() -> int:
    df = pd.read_parquet(OUT_MANIFEST)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[FAIL] missing columns: {missing}")
        return 2
    per_class = df.groupby("class_label").size().to_dict()
    print("[OK] counts per class:", per_class)
    bad = df[df["source"].eq("WSI") & (~df["has_wsi"])]
    if len(bad):
        print(f"[WARN] missing WSI rows: {len(bad)}")
    if "slide_id" in df.columns:
        dup = df[df["slide_id"].notna()].duplicated(subset=["patient_id","slide_id"], keep=False)
        if dup.any():
            print(f"[WARN] duplicated (patient,slide): {dup.sum()}")
    print("[OK] manifest validation")
    return 0

if __name__ == "__main__":
    sys.exit(main())
PY

# ──────────────────────────────────────────────────────────────────────────────
# step_05_datacard.py
# ──────────────────────────────────────────────────────────────────────────────
cat > "$HOME_ROOT/src/ingest/step_05_datacard.py" <<'PY'
from __future__ import annotations
from pathlib import Path
import pandas as pd, yaml
from .config_ingest import OUT_MANIFEST, OUT_DATACARD

def main() -> Path:
    df = pd.read_parquet(OUT_MANIFEST)
    summary = {
        "rows": int(df.shape[0]),
        "classes": {c:int(v) for c,v in df.groupby("class_label").size().items()},
        "by_source": {c:int(v) for c,v in df.groupby("source").size().items()},
        "qc_flags": {k:int(v) for k,v in df["notes"].value_counts(dropna=True).items()},
        "columns": list(df.columns),
    }
    OUT_DATACARD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DATACARD.open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return OUT_DATACARD

if __name__ == "__main__":
    p = main()
    print(f"[OK] datacard → {p}")
PY

# ──────────────────────────────────────────────────────────────────────────────
# step_06_smoketest.py
# ──────────────────────────────────────────────────────────────────────────────
cat > "$HOME_ROOT/src/ingest/step_06_smoketest.py" <<'PY'
from __future__ import annotations
import sys, pandas as pd
from .config_ingest import OUT_MANIFEST

def main() -> int:
    try:
        df = pd.read_parquet(OUT_MANIFEST)
        assert df.shape[0] > 0, "empty manifest"
        assert df["patient_id"].notna().all(), "null patient_id"
        print(df.sample(min(5, len(df))))
        print("[OK] smoketest")
        return 0
    except Exception as e:
        print("[FAIL] smoketest:", e)
        return 3

if __name__ == "__main__":
    sys.exit(main())
PY

# ──────────────────────────────────────────────────────────────────────────────
# run_all_ingest.py (orchestratore)
# ──────────────────────────────────────────────────────────────────────────────
cat > "$HOME_ROOT/src/ingest/run_all_ingest.py" <<'PY'
"""One-shot orchestrator: esegue tutti gli step in sequenza.
Usage: python -m ingest.run_all_ingest
"""
from __future__ import annotations
import sys
from . import step_01_scan as s1
from . import step_03_build_manifest as s3
from . import step_04_validate as s4
from . import step_05_datacard as s5
from . import step_06_smoketest as s6

def main() -> int:
    print("[1/5] scan files…")
    lookup = s1.main()

    print("[2/5] build manifest…")
    df = s3.main(lookup)
    print(f"[INFO] manifest rows: {len(df)}")

    print("[3/5] validate manifest…")
    rc = s4.main()
    if rc != 0:
        return rc

    print("[4/5] write data card…")
    s5.main()

    print("[5/5] smoketest…")
    return s6.main()

if __name__ == "__main__":
    sys.exit(main())
PY

# ──────────────────────────────────────────────────────────────────────────────
# SLURM: run_all_ingest.sbatch
# ──────────────────────────────────────────────────────────────────────────────
cat > "$HOME_ROOT/slurm/ingest/run_all_ingest.sbatch" <<'SB'
#!/bin/bash
#SBATCH --job-name=ingest_all
#SBATCH --partition=cpu_sapphire
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl
#SBATCH --output=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.out
#SBATCH --error=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.err

module purge
module load python/3.10
source $HOME/.venvs/rcc/bin/activate || true

export PYTHONPATH=/home/mla_group_01/rcc-ssrl/src:$PYTHONPATH
export SCRATCH=/beegfs-scratch/mla_group_01/rcc-ssrl

python -m ingest.run_all_ingest
SB

echo "[OK] Files created."
echo
echo "▶ Esecuzione interattiva:"
echo "   export PYTHONPATH=/home/mla_group_01/rcc-ssrl/src:\$PYTHONPATH"
echo "   python -m ingest.run_all_ingest"
echo
echo "▶ Esecuzione batch (SLURM):"
echo "   sbatch /home/mla_group_01/rcc-ssrl/slurm/ingest/run_all_ingest.sbatch"
