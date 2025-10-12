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
