#!/usr/bin/env python3
"""
Builds slides.parquet and rcc_dataset_stats.json by joining:
- metadata CSV (WSI ↔ XML/ROI ↔ patient)
- inventory (WSI technical metadata), optional but recommended

Output in OUTPUT_DIR:
  - slides.parquet         (normalized schema, 1 row = 1 WSI)
  - slides.csv             (convenient for human QA)
  - rcc_dataset_stats.json (counts and quick checks)
"""
import argparse, json, os, re, sys, hashlib
from pathlib import Path
import pandas as pd

def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[/\\]", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_").lower()

def hash_id(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()

def load_metadata(meta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_path)
    # Normalize expected columns
    required = ["subtype","patient_id","wsi_filename","annotation_xml","num_annotations","roi_files","num_rois","source_dir"]
    for c in required:
        if c not in df.columns:
            # Allow missing columns but create them empty
            if c in ("annotation_xml","roi_files","source_dir"):
                df[c] = ""
            elif c in ("num_annotations","num_rois"):
                df[c] = 0
            else:
                raise ValueError(f"Missing required column: {c}")
    # Coerce types
    df["subtype"]       = df["subtype"].astype(str)
    df["patient_id"]    = df["patient_id"].astype(str)
    df["wsi_filename"]  = df["wsi_filename"].astype(str)
    df["annotation_xml"]= df["annotation_xml"].astype(str)
    df["num_annotations"]= pd.to_numeric(df["num_annotations"], errors="coerce").fillna(0).astype(int)
    df["roi_files"]     = df["roi_files"].astype(str)
    df["num_rois"]      = pd.to_numeric(df["num_rois"], errors="coerce").fillna(0).astype(int)
    df["source_dir"]    = df["source_dir"].astype(str)
    # Basic derived fields
    df["wsi_basename"]  = df["wsi_filename"].apply(lambda x: Path(x).name)
    df["class_label"]   = df["subtype"]  # semantic alias
    # Flag ROI 'not tumour' (if advanced columns present)
    for c in ("xml_roi_total","xml_roi_tumor","xml_roi_non_tumor"):
        if c not in df.columns: df[c] = 0
    df["has_not_tumour"] = (df["xml_roi_non_tumor"] > 0).astype(bool)
    return df

def load_inventory(inv_path: Path) -> pd.DataFrame:
    if not inv_path or not inv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(inv_path)
    # Align keys
    if "rel_path" in df.columns:
        df["wsi_basename"] = df["rel_path"].apply(lambda p: Path(str(p)).name)
    else:
        df["wsi_basename"] = df["slide_path"].apply(lambda p: Path(str(p)).name)
    # Useful types
    for col in ("vendor","ext","compression","read_backend","class_hint"):
        if col in df.columns: df[col] = df[col].astype(str)
    for col in ("width0","height0","level_count"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("mpp_x","mpp_y"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_parquet(meta_df: pd.DataFrame, inv_df: pd.DataFrame) -> pd.DataFrame:
    # Merge on basename
    if not inv_df.empty:
        keep_cols = [
          "slide_path","rel_path","ext","vendor","width0","height0","level_count",
          "level_downsamples","mpp_x","mpp_y","objective_power","compression",
          "xml_present","xml_roi_total","xml_roi_tumor","xml_roi_non_tumor","class_hint"
        ]
        keep_cols = [c for c in keep_cols if c in inv_df.columns]
        merged = meta_df.merge(inv_df[["wsi_basename"]+keep_cols], how="left", on="wsi_basename")
        # If inventory brings better xml counts, override
        for c in ("xml_roi_total","xml_roi_tumor","xml_roi_non_tumor"):
            if c in merged.columns:
                merged[c] = merged[c].fillna(0).astype(int)
        if "xml_roi_non_tumor" in merged.columns:
            merged["has_not_tumour"] = (merged["xml_roi_non_tumor"] > 0)
    else:
        merged = meta_df.copy()
        # Synthesize path/rel_path from source_dir
        merged["rel_path"] = merged.apply(lambda r: f"{r['source_dir'].strip('/')}/{r['wsi_filename']}".strip("/"), axis=1)
        merged["slide_path"] = merged["rel_path"]  # can be made absolute downstream

    # IDs and useful keys
    merged["slide_id"]  = merged.apply(lambda r: slugify(f"{r['class_label']}_{r['patient_id']}_{Path(r['wsi_filename']).stem}"), axis=1)
    merged["record_id"] = merged.apply(lambda r: hash_id(r["slide_id"], r["class_label"]), axis=1)
    merged["subset"]    = merged.get("subset", pd.Series(["unassigned"]*len(merged)))

    # Normalize text fields
    for c in ("annotation_xml","roi_files","source_dir","rel_path","slide_path","level_downsamples","compression","vendor","ext","objective_power"):
        if c in merged.columns:
            merged[c] = merged[c].fillna("").astype(str)

    # Recommended column order
    ordered = [
        "record_id", # "slide_id",  # removed
        "patient_id","class_label", # "subset",  # removed
        # "slide_path",  # rimosso
        "rel_path","wsi_filename","subtype","source_dir",
        "ext","vendor","objective_power","mpp_x","mpp_y",
        "width0","height0","level_count","level_downsamples","compression",
        "annotation_xml", # "num_annotations",  # removed
        "roi_files","num_rois",
        "xml_roi_total","xml_roi_tumor", # "xml_roi_non_tumor",  # removed
        # "has_not_tumour",  # removed
        "wsi_basename","class_hint"
    ]
    # Exclude the required columns from output
    exclude_cols = {"subset", "slide_path", "slide_id", "num_annotations", "xml_roi_non_tumor", "has_not_tumour"}
    cols = [c for c in ordered if c in merged.columns and c not in exclude_cols] + \
           [c for c in merged.columns if c not in ordered and c not in exclude_cols]
    return merged[cols].copy()

def make_stats(df: pd.DataFrame) -> dict:
    stats = {
        "n_rows": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "by_class": df["class_label"].value_counts().to_dict(),
        "patients_by_class": df.groupby("class_label")["patient_id"].nunique().to_dict(),
        "by_subset": df["subset"].value_counts().to_dict(),
        "by_class_subset": df.groupby(["class_label","subset"]).size().to_dict(),
        "has_not_tumour_by_class": df.groupby("class_label")["has_not_tumour"].sum().astype(int).to_dict(),
    }
    for col in ("vendor","ext"):
        if col in df.columns:
            stats[f"by_{col}"] = df[col].value_counts().to_dict()
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=str, required=False,
                    default="/home/mla_group_01/configs/rcc_metadata.csv")
    ap.add_argument("--inventory", type=str, required=False,
                    default="/home/mla_group_01/reports/0_phase/wsi_inventory.csv")
    ap.add_argument("--output-dir", type=str, required=False,
                    default="/home/mla_group_01/reports/02_parquet")
    ap.add_argument("--csv-also", action="store_true", help="Also write slides.csv")
    args = ap.parse_args()

    meta_path = Path(args.metadata)
    inv_path  = Path(args.inventory)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        print(f"[ERROR] metadata CSV not found: {meta_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Reading metadata:  {meta_path}")
    meta_df = load_metadata(meta_path)

    inv_df = pd.DataFrame()
    if inv_path.exists():
        print(f"[INFO] Reading inventory: {inv_path}")
        inv_df = load_inventory(inv_path)
    else:
        print(f"[WARN] Inventory missing: {inv_path} (proceeding with metadata only)")

    df = build_parquet(meta_df, inv_df)

    # Write Parquet
    pq_path = out_dir / "slides.parquet"
    csv_path = out_dir / "slides.csv"
    print(f"[INFO] Writing: {pq_path}")
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"[WARN] pyarrow/fastparquet not available ({e}); saving CSV fallback")
        df.to_csv(csv_path, index=False)

    if args.csv_also:
        print(f"[INFO] Writing (also) CSV: {csv_path}")
        df.to_csv(csv_path, index=False)

    # Stats JSON
    stats = make_stats(df)
    with (out_dir / "rcc_dataset_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Righe: {len(df)}  | Pazienti: {df['patient_id'].nunique()}")
    print(f"[OK] Stats: {out_dir / 'rcc_dataset_stats.json'}")
    print("[DONE]")
if __name__ == "__main__":
    main()
