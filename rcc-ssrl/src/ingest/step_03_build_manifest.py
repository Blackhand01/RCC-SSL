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
