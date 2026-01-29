#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrichment metadata.csv -> rcc_metadata.csv (one row for each WSI)
- NO path/slide_id in the output field(only filename/dir as in input)
- Only byte/propriety of the WSI (no XML)
-Sum ROI da XML: xml_roi_tumor, xml_roi_not_tumor, xml_roi_total
"""
import argparse, os, re, sys
from pathlib import Path
import pandas as pd
from typing import Optional

# best-effort deps
try:
    import openslide
except Exception:
    openslide = None
try:
    import tifffile
except Exception:
    tifffile = None
try:
    import lxml.etree as ET
except Exception:
    ET = None

SUPPORTED_WSI = {".svs",".tif",".tiff",".scn",".ndpi",".mrxs",".bif",".svslide",".czi"}

TUMOR_KEYS = ("tumor","tumour","carcinom","neoplas","malignan")
NOT_TUMOR_KEYS = (
    "not","non","normal","parenchyma","parenchima","stroma","stromal",
    "fibro","fiber","fibrosis","necrosi","necrosis","necrotic","inflamm","benign"
)

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[_\-.,:/\s]+", " ", s)
    return s.strip()

def xml_roi_counts(xml_path: Path):
    """Conta ROI tumore vs not-tumour (Aperio: su Name, ASAP: su PartOfGroup)."""
    if (ET is None) or (not xml_path or not xml_path.is_file()):
        return (0,0,0)
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return (0,0,0)

    tumor = not_tumor = total = 0
    for a in root.findall(".//Annotation"):
        name = norm(a.get("Name") or a.get("name") or "")
        group = norm(a.get("PartOfGroup") or a.get("partofgroup") or "")
        # ASAP: region = <Coordinate> (count all), Aperio: region = <Region>
        regions = a.findall(".//Region") or a.findall(".//region")
        if not regions:
            regions = a.findall(".//Coordinates") or a.findall(".//Coordinate")
        nreg = len(regions) if regions else 1  # fallback: at least 1 if annotation exists
        total += nreg

        if group and any(k in group for k in NOT_TUMOR_KEYS):
            not_tumor += nreg
        elif group and any(k in group for k in TUMOR_KEYS):
            tumor += nreg
        # Aperio
        elif ("non" in name or "not" in name) and ("tumor" in name or "tumour" in name):
            not_tumor += nreg
        elif any(k in name for k in NOT_TUMOR_KEYS) and not any(k in name for k in TUMOR_KEYS):
            not_tumor += nreg
        elif any(k in name for k in TUMOR_KEYS):
            tumor += nreg

    return (tumor, not_tumor, total)

def safe_wsi_props(wsi_path: Path):
    """Return propriety ONLY of the WSI (size,dims,levels,mpp,vendor,objective)."""
    size_bytes = wsi_path.stat().st_size if wsi_path.exists() else None
    vendor = width0 = height0 = level_count = mpp_x = mpp_y = objective = None

    # Tentativo openslide
    if openslide is not None and wsi_path.suffix.lower() in SUPPORTED_WSI:
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            width0, height0 = slide.dimensions
            level_count = slide.level_count
            prop = slide.properties
            vendor = prop.get("openslide.vendor")
            mpp_x = prop.get("openslide.mpp-x")
            mpp_y = prop.get("openslide.mpp-y")
            objective = prop.get("openslide.objective-power") or prop.get("aperio.AppMag")
            try: slide.close()
            except Exception: pass
        except Exception:
            pass

    # Fallback per TIFF 
    if (width0 is None or height0 is None) and tifffile is not None and wsi_path.suffix.lower() in {".tif",".tiff"}:
        try:
            with tifffile.TiffFile(str(wsi_path)) as tf:
                page = tf.pages[0]
                sh = getattr(page, "shape", None)
                if sh is not None:
                    if len(sh)==2:
                        height0, width0 = int(sh[0]), int(sh[1])
                    elif len(sh)>=3:
                        height0, width0 = int(sh[-2]), int(sh[-1])
        except Exception:
            pass

    return dict(
        wsi_size_bytes=size_bytes, vendor=vendor, width0=width0, height0=height0,
        level_count=level_count, mpp_x=mpp_x, mpp_y=mpp_y, objective_power=objective
    )

def find_xml_for_row(raw_root: Path, wsi_path: Path, xml_name: Optional[str]):
    """Risolvi il path XML: 1) stesso dir della WSI se esiste, 2) ricerca globale by name."""
    if not xml_name:
        cand = wsi_path.with_suffix(".xml")
        return cand if cand.exists() else None
    cand = (wsi_path.parent / xml_name)
    if cand.exists():
        return cand
    hits = list(raw_root.rglob(xml_name))
    return hits[0] if hits else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--metadata-csv", required=True)
    args = ap.parse_args()

    raw_root = Path(args.raw_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    meta_csv = Path(args.metadata_csv).resolve()

    df = pd.read_csv(meta_csv)
    base_cols = ["subtype","patient_id","wsi_filename","annotation_xml","num_annotations","roi_files","num_rois","source_dir"]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] metadata.csv missing columns: {missing}")

    out_rows = []

    for _, r in df.iterrows():
        subtype     = r.get("subtype")
        patient_id  = r.get("patient_id")
        wsi_name    = r.get("wsi_filename")
        xml_name    = r.get("annotation_xml")
        roi_files   = r.get("roi_files")
        num_rois    = r.get("num_rois")
        src_dir     = r.get("source_dir")

        wsi_path = (raw_root / str(src_dir) / str(wsi_name)).resolve()
        xml_path = find_xml_for_row(raw_root, wsi_path, str(xml_name) if pd.notna(xml_name) else None)

        props = dict.fromkeys(["wsi_size_bytes","vendor","width0","height0","level_count","mpp_x","mpp_y","objective_power"])
        if wsi_path.exists():
            props.update(safe_wsi_props(wsi_path))
        else:
            props["wsi_size_bytes"] = None

        # ROI counts da XML
        tumor, not_tumor, total = xml_roi_counts(xml_path) if xml_path else (0,0,0)

        out_rows.append({
            # --- Principal keys ---
            "subtype": subtype,
            "patient_id": patient_id,
            "wsi_filename": wsi_name,
            "annotation_xml": xml_name if pd.notna(xml_name) else "",
            "roi_files": roi_files if pd.notna(roi_files) else "",
            "num_rois": int(num_rois) if pd.notna(num_rois) else 0,
            "source_dir": src_dir,

            # --- ONLY WSI (no path, no xml bytes) ---
            **props,

            # --- XML derived stats ---
            "xml_roi_tumor": int(tumor),
            "xml_roi_not_tumor": int(not_tumor),
            "xml_roi_total": int(total),
        })

    out = pd.DataFrame(out_rows)

    # Final column order
    cols = [
        "subtype","patient_id","wsi_filename","annotation_xml","roi_files","num_rois","source_dir",
        "wsi_size_bytes","vendor","width0","height0","level_count","mpp_x","mpp_y","objective_power",
        "xml_roi_tumor","xml_roi_not_tumor","xml_roi_total"
    ]
    out = out[cols]

    out_csv = report_dir / "rcc_metadata.csv"
    tmp = out_csv.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(out_csv)
    print("[OK] written", out_csv)

if __name__ == "__main__":
    main()
