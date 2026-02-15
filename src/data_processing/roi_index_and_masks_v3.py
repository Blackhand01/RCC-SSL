#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3: XML indexing and mask saving with ATOMIC writing in rcc_masks_v3
Fix: tmp in same dir, flush+fsync, os.replace, retry on shared FS.
"""
import os, io, json, time, math, argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np, pandas as pd, yaml
from lxml import etree

# Limit BLAS threads
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_MAX_THREADS","1")

# ----------------------- util -----------------------
def norm(s): return (s or "").strip().lower()

def atomic_save_npz(dst: Path, arrays: dict, max_retries: int = 4, sleep_base: float = 0.15) -> int:
    """
    Write dst (.npz) atomically: tmp in SAME dir, flush+fsync, os.replace.
    Returns final file size in bytes.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp to avoid collisions
    tag = f"{os.getpid()}_{int(time.time()*1e6)}"
    tmp = dst.parent / f"{dst.name}.{tag}.tmp"
    attempt = 0
    while True:
        try:
            with open(tmp, "wb") as f:
                # Use savez_compressed on explicit file handle
                np.savez_compressed(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
            # Atomic rename in same partition
            os.replace(tmp, dst)
            sz = dst.stat().st_size
            return sz
        except Exception as e:
            attempt += 1
            # Cleanup tmp if exists
            try:
                if tmp.exists(): tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt > max_retries:
                raise
            time.sleep(sleep_base * (2 ** (attempt-1)))

def choose_level_safe(slide, target_mpp: float, patch_px: int, max_side: int, min_side: int) -> int:
    import openslide
    L = slide.level_count
    dims = [slide.level_dimensions[i] for i in range(L)]
    downs = [slide.level_downsamples[i] for i in range(L)]
    try:
        mpp0 = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "0") or 0.0)
    except Exception:
        mpp0 = 0.0
    lvl = None
    if mpp0 > 0 and target_mpp > 0:
        levels = [mpp0 * d for d in downs]
        lvl = int(np.argmin([abs(x - target_mpp) for x in levels]))
    if lvl is None:
        lvl = L - 1
        for i, (W,H) in enumerate(dims):
            if max(W,H) <= max_side and min(W,H) >= max(min_side, 2*patch_px):
                lvl = i
                break
    while max(dims[lvl]) > max_side and lvl < L-1: lvl += 1
    while min(dims[lvl]) < max(min_side, 2*patch_px) and lvl > 0: lvl -= 1
    return int(lvl)

def xml_regions(xml_path: Path) -> List[dict]:
    """
    Returns list of dicts with level 0 coordinates:
      { label_text: str, points: [(x,y),...], meta:{source:"asap|aperio|leica", name, group, type} }
    Supports:
      - ASAP: <Annotation ... PartOfGroup="necrosis"> <Coordinates><Coordinate .../></Coordinates></Annotation>
      - Aperio-like: <Region Text="tumor"><Vertex X=... Y=.../></Region>
      - Leica-like:  <Coordinates><Coordinate X=... Y=.../></Coordinates>
    """
    root = etree.parse(str(xml_path)).getroot()
    regions = []

    # ---- ASAP ----
    # Example: <Annotation Name="Annotation 0" Type="Spline" PartOfGroup="necrosis" ...>
    for ann in root.xpath(".//Annotation"):
        group = ann.get("PartOfGroup") or ""
        name  = ann.get("Name") or ""
        atype = ann.get("Type") or ""
        coords_nodes = ann.xpath(".//Coordinates")
        for cset in coords_nodes:
            pts=[]
            for v in cset.xpath(".//Coordinate"):
                x=v.get("X"); y=v.get("Y")
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
            if len(pts) >= 3:
                regions.append({
                    "label_text": group if group else name,   # preferisci PartOfGroup
                    "points": pts,
                    "meta": {"source":"asap","name":name,"group":group,"type":atype}
                })

    # ---- Aperio-like: <Region Text="..."><Vertex .../></Region> ----
    for reg in root.xpath(".//Region"):
        text = reg.get("Text") or reg.get("Description") or reg.get("Id") or ""
        vs = reg.xpath(".//Vertex")
        pts = [(float(v.get("X")), float(v.get("Y"))) for v in vs if v.get("X") and v.get("Y")]
        if len(pts) >= 3:
            regions.append({
                "label_text": text,
                "points": pts,
                "meta": {"source":"aperio"}
            })

    # ---- Leica-like fallback: <Coordinates><Coordinate .../></Coordinates> without label ----
    # NB: if it reaches here without label, map_group will return None and will be ignored
    if not regions:
        for poly in root.xpath(".//Coordinates"):
            pts=[]
            for v in poly.xpath(".//Coordinate"):
                x=v.get("X"); y=v.get("Y")
                if x and y: pts.append((float(x), float(y)))
            if len(pts)>=3:
                regions.append({
                    "label_text": "",
                    "points": pts,
                    "meta": {"source":"leica"}
                })
    return regions

def map_group(label_map: dict, label_text: str) -> Optional[str]:
    lt = norm(label_text)
    # Match "contains" against extended lists (ita/eng)
    for k in label_map.get("tumor", []):
        if norm(k) in lt and k:
            return "tumor"
    for k in label_map.get("not_tumor", []):
        if norm(k) in lt and k:
            return "not_tumor"
    return None

def rasterize(polys, shape_wh: Tuple[int,int]) -> np.ndarray:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from skimage.draw import polygon as sk_polygon
    W,H = shape_wh
    mask = np.zeros((H, W), dtype=np.uint8)
    if not polys: return mask
    try:
        U = unary_union(polys)
        if U.is_empty: return mask
        geoms = [U] if U.geom_type == "Polygon" else list(U.geoms)
        for poly in geoms:
            if poly.is_empty: continue
            xs, ys = poly.exterior.coords.xy
            rr, cc = sk_polygon(np.asarray(ys), np.asarray(xs), (H, W))
            mask[rr, cc] = 1
            for hole in poly.interiors:
                xs, ys = hole.coords.xy
                rr, cc = sk_polygon(np.asarray(ys), np.asarray(xs), (H, W))
                mask[rr, cc] = 0
    except Exception:
        pass
    return mask

def resolve_xml_path(xml_name: str, cls_label: str, rel_path: Path, roots: dict) -> Optional[Path]:
    cands = [rel_path.parent / xml_name]
    cands += [Path(roots.get("pre_ccRCC_xml_root",""))/xml_name,
              Path(roots.get("pre_pRCC_xml_root",""))/xml_name]
    if cls_label == "ccRCC":
        cands.append(Path(roots.get("ccRCC_xml_root",""))/xml_name)
    elif cls_label == "pRCC":
        cands.append(Path(roots.get("pRCC_xml_root",""))/xml_name)
    cands += [Path(roots.get("ccRCC_xml_root",""))/xml_name,
              Path(roots.get("pRCC_xml_root",""))/xml_name]
    for p in cands:
        if p and p.exists(): return p
    return None

# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="Append/skip already present in masks_index.jsonl")
    ap.add_argument("--rewrite-xml-masks", action="store_true", help="Regenerate even if .npz exist")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    slides_pq = Path(cfg["paths"]["parquet"])
    wsi_root  = Path(cfg["paths"]["wsi_root"])
    out_masks = Path(cfg["paths"]["out_masks_v3"]); out_masks.mkdir(parents=True, exist_ok=True)
    ann_roots = cfg["paths"]["annotations"]
    xml_map   = cfg["xml_label_map"]
    target_mpp = float(cfg["target_mpp"])
    patch_px   = int(cfg["patch_size_px"])
    max_side   = int(cfg["masks"]["max_mask_side"])
    min_side   = int(cfg["masks"]["min_mask_side"])
    progress_every = int(cfg.get("logging", {}).get("progress_every", 20))

    df = pd.read_parquet(slides_pq)

    index_path = out_masks/"masks_index.jsonl"
    seen_ids = set()
    if args.resume and index_path.exists():
        for line in index_path.read_text().splitlines():
            try: seen_ids.add(json.loads(line)["record_id"])
            except: pass
        mode="a"
    else:
        mode="w"

    processed=written=skipped=errors=0
    with index_path.open(mode) as fout:
        for _, r in df.iterrows():
            record_id = str(r["record_id"])
            pid       = str(r["patient_id"])
            cls       = str(r["class_label"])
            rel       = Path(str(r["rel_path"]))
            xml_name  = str(r.get("annotation_xml","") or "")

            if cls not in ("ccRCC","pRCC"):
                # For v3: we index only XML (CHROMO/ONCO ROI not touched here)
                continue

            if args.resume and (record_id in seen_ids) and not args.rewrite_xml_masks:
                skipped += 1; continue

            wsi_path = (wsi_root/rel)
            try:
                import openslide
                slide = openslide.OpenSlide(str(wsi_path))
            except Exception:
                errors += 1; continue

            if not xml_name:
                slide.close(); skipped += 1; continue

            xml_path = resolve_xml_path(xml_name, cls, wsi_path, ann_roots)
            if not xml_path:
                slide.close(); skipped += 1; continue

            try:
                level = choose_level_safe(slide, target_mpp, patch_px, max_side=max_side, min_side=min_side)
                W,H = slide.level_dimensions[level]
                ds  = slide.level_downsamples[level]

                regs = xml_regions(xml_path)
                from shapely.geometry import Polygon
                polys_t, polys_n = [], []
                for rr in regs:
                    grp = map_group(xml_map, rr["label_text"])
                    if grp is None: 
                        continue
                    pts_lvl = [(x/ds, y/ds) for (x,y) in rr["points"]]
                    try:
                        P = Polygon(pts_lvl).buffer(0)
                        if not P.is_valid or P.is_empty: 
                            continue
                    except Exception:
                        continue
                    (polys_t if grp=="tumor" else polys_n).append(P)

                tumor_mask = rasterize(polys_t, (W,H))
                not_mask   = rasterize(polys_n, (W,H))

                dst_dir = out_masks/record_id; dst_dir.mkdir(parents=True, exist_ok=True)

                sz_t = atomic_save_npz(
                    dst_dir/f"tumor_L{level}.npz",
                    {"mask": tumor_mask, "level": int(level), "width": int(W), "height": int(H), "ds": float(ds)}
                )
                sz_n = atomic_save_npz(
                    dst_dir/f"not_tumor_L{level}.npz",
                    {"mask": not_mask, "level": int(level), "width": int(W), "height": int(H), "ds": float(ds)}
                )

                meta = {
                    "kind":"xml_masks", "record_id":record_id, "patient_id":pid,
                    "class_label":cls, "wsi_rel_path":str(rel), "level":int(level),
                    "npz_sizes":{"tumor": int(sz_t), "not_tumor": int(sz_n)}
                }
                # Avoid duplicates in resume without rewrite
                if (not args.resume) or args.rewrite_xml_masks or (record_id not in seen_ids):
                    fout.write(json.dumps(meta)+"\n"); written += 1

            except Exception:
                errors += 1
            finally:
                slide.close()
                processed += 1
                if processed % max(1,progress_every) == 0:
                    print(f"[PROGRESS] proc={processed} writ={written} skip={skipped} err={errors}")

    print(f"[DONE] index: {index_path} | proc={processed} writ={written} skip={skipped} err={errors}")

if __name__ == "__main__":
    main()
