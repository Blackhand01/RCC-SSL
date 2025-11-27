#!/usr/bin/env python3
import argparse, json, yaml, gc, os
from math import ceil
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List
import numpy as np

# Limita thread
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def read_jsonl_lines(path: Path):
    if not path.exists(): return
    with path.open() as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: yield json.loads(line)
            except: pass

def load_npz_mask(p: Path) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float], Optional[Tuple[int,int]]]:
    try:
        if not p.exists(): return None, None, None, None
        d = np.load(str(p), allow_pickle=False)
        m = d["mask"].astype(np.uint8)
        lvl = int(d.get("level", 0))
        ds  = float(d.get("ds", 1.0))
        H, W = m.shape
        return m, lvl, ds, (W, H)
    except Exception:
        return None, None, None, None

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys = np.any(mask, axis=1)
    xs = np.any(mask, axis=0)
    if not ys.any() or not xs.any():
        return None
    y_idx = np.where(ys)[0]; x_idx = np.where(xs)[0]
    return int(x_idx[0]), int(y_idx[0]), int(x_idx[-1]), int(y_idx[-1])

def downscale_mask_binary_power2(mask: np.ndarray, target_px: int) -> Tuple[np.ndarray, int]:
    """Riduci finché H*W <= target_px via max-pooling 2x."""
    H, W = mask.shape
    factor = 1
    while H*W > target_px and H >= 2 and W >= 2:
        H2 = (H // 2) * 2
        W2 = (W // 2) * 2
        m = mask[:H2, :W2]
        m = m.reshape(H2//2, 2, W2//2, 2).max(axis=(1,3)).astype(np.uint8)
        mask = m
        H, W = mask.shape
        factor *= 2
    return mask, factor

def sample_center_in_mask(mask: np.ndarray, bbox: Tuple[int,int,int,int],
                          patch_eff: int, need: int,
                          rng: np.random.Generator, max_trials: int) -> List[Tuple[int,int]]:
    """Sceglie (x,y) top-left con centro nel ROI."""
    if need <= 0: return []
    x0,y0,x1,y1 = bbox
    H, W = mask.shape
    xmin = max(x0, 0); ymin = max(y0, 0)
    xmax = min(x1 - patch_eff + 1, W - patch_eff)
    ymax = min(y1 - patch_eff + 1, H - patch_eff)
    if xmax < xmin or ymax < ymin: return []
    coords=[]; trials=0
    cx_off = patch_eff // 2; cy_off = patch_eff // 2
    while len(coords) < need and trials < max_trials:
        x = int(rng.integers(xmin, xmax+1))
        y = int(rng.integers(ymin, ymax+1))
        cx = x + cx_off; cy = y + cy_off
        if mask[cy, cx] != 0:
            coords.append((x, y))
        trials += 1
    return coords

# ---------- helper: path resolver per ROI ----------
def resolve_roi_path(rp: str, cls: str, cfg: dict) -> str:
    """
    Rende 'rp' in path esistente sul FS. Prova diverse radici e sottocartelle comuni.
    Ritorna stringa vuota se non risolto.
    """
    p = Path(rp)
    if p.exists():
        return str(p)

    ann = cfg["paths"]["annotations"]
    chromo_root = Path(ann["chromo_roi_root"])
    onco_root   = Path(ann["onco_roi_root"])
    fname = p.name

    roots = [chromo_root] if cls == "CHROMO" else [onco_root]
    candidates = []
    for root in roots:
        candidates.extend([
            root / fname,
            root / Path(rp),
            root / "tif" / fname,
            root / "TIFF" / fname,
            root / "TIF" / fname,
        ])
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return ""

def get_image_size_any(path_str: str) -> Optional[Tuple[int,int]]:
    """
    Restituisce (W,H) usando PIL se possibile, altrimenti OpenSlide per WSI (.svs/.scn/.ndpi/.mrxs).
    Ritorna None se non apribile.
    """
    path = Path(path_str)
    # 1) Prova PIL
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path_str) as im:
            return im.size  # (W,H)
    except Exception:
        pass
    # 2) Prova OpenSlide se è un formato WSI noto
    if path.suffix.lower() in {".svs", ".scn", ".ndpi", ".mrxs"}:
        try:
            import openslide
            osr = openslide.OpenSlide(path_str)
            W, H = osr.dimensions
            osr.close()
            return (W, H)
        except Exception:
            return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset-only", choices=["train","val","test"], default=None)
    ap.add_argument("--only-patient", default=None)
    ap.add_argument("--source", choices=["both","xml","roi"], default="both",
                    help="Genera da xml_masks (wsi), da roi_svs (roi), o entrambi.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rng = np.random.default_rng(int(cfg.get("generation", {}).get("seed", 1337)))

    # parametri
    patch = int(cfg["patch_size_px"])
    gen   = cfg.get("generation", {})
    budgets = gen.get("budget_total", {"train":300000,"val":60000,"test":60000})
    slack   = float(gen.get("slack_per_patient", 1.30))
    per_roi_cap = int(gen.get("per_roi_cap", 8000))
    per_wsi_cap = int(gen.get("per_wsi_cap", 8000))
    max_trials_factor = int(gen.get("max_trials_factor", 30))
    max_trials = max(100, max_trials_factor * 50)
    max_mask_pixels = int(gen.get("max_mask_pixels", 40_000_000))

    # path
    folds      = json.loads(Path(cfg["split"]["folds_json"]).read_text())
    out_masks  = Path(cfg["paths"]["out_masks"])
    out_cand   = ensure_dir(Path(cfg["paths"]["out_candidates"]))
    idx_path   = out_masks / "masks_index.jsonl"
    wsi_root   = Path(cfg["paths"]["wsi_root"])
    assert idx_path.exists(), f"Missing {idx_path}"

    wanted = [args.subset_only] if args.subset_only else ["train","val","test"]
    fouts = {s: (out_cand/f"{s}.jsonl").open("a") for s in wanted}

    # resume counters (totale + per sorgente)
    cnt_patient     = defaultdict(int)                # (subset,pid) -> n totale
    cnt_patient_src = defaultdict(int)                # (subset,pid,origin) -> n  origin in {"wsi","roi"}
    cnt_wsi         = defaultdict(int)                # record_id (xml_masks)
    cnt_roi         = defaultdict(int)                # (record_id, roi_path_resolved)

    pid2subset = {p:s for s,ps in folds["patients"].items() for p in ps}

    def read_prev(subset: str):
        prev = out_cand/f"{subset}.jsonl"
        for j in read_jsonl_lines(prev):
            pid = j.get("patient_id"); rid = j.get("record_id")
            origin = j.get("origin")
            cnt_patient[(subset, pid)] += 1
            if origin=="wsi":
                cnt_wsi[rid] += 1
                cnt_patient_src[(subset, pid, "wsi")] += 1
            elif origin=="roi":
                rp_res = j.get("source_rel_path","")
                cnt_roi[(rid, rp_res)] += 1
                cnt_patient_src[(subset, pid, "roi")] += 1

    for s in wanted: read_prev(s)

    pats_by_subset = {s:set(ps) for s,ps in folds["patients"].items()}
    per_patient_cap = {}
    for s in ("train","val","test"):
        n = max(1, len(pats_by_subset.get(s,[])))
        per_patient_cap[s] = int(ceil(budgets.get(s,0)/n) * slack)

    # raccogli per paziente
    by_patient = defaultdict(lambda: {"subset": None, "xml": [], "roi": []})
    with idx_path.open() as fin:
        for line in fin:
            line=line.strip()
            if not line: continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            pid = j.get("patient_id")
            if not pid: continue
            subset = pid2subset.get(pid)
            if subset not in wanted: continue
            if args.only_patient and pid != args.only_patient: continue
            by_patient[pid]["subset"] = subset
            if j.get("kind")=="xml_masks":
                by_patient[pid]["xml"].append(j)
            else:
                by_patient[pid]["roi"].append(j)

    # processa pazienti; priorità xml (cc/p) poi ROI
    order = sorted(by_patient.keys(), key=lambda p: (len(by_patient[p]["xml"])==0, p))

    written_total = 0
    def out_write(s, obj):
        nonlocal written_total
        fouts[s].write(json.dumps(obj)+"\n"); written_total += 1

    for pid in order:
        subset = by_patient[pid]["subset"]
        capP   = per_patient_cap[subset]

        xmls = by_patient[pid]["xml"]
        rois = by_patient[pid]["roi"]

        do_xml = (args.source in ("both","xml")) and xmls and (cnt_patient_src[(subset,pid,"wsi")] < capP)
        do_roi = (args.source in ("both","roi")) and rois and (cnt_patient_src[(subset,pid,"roi")] < capP)

        if args.debug:
            print(f"[PAT] {pid} subset={subset} cap={capP} prev_total={cnt_patient[(subset,pid)]} "
                  f"prev_wsi={cnt_patient_src[(subset,pid,'wsi')]} prev_roi={cnt_patient_src[(subset,pid,'roi')]} "
                  f"do_xml={do_xml} do_roi={do_roi}")

        # ----------- ccRCC / pRCC (xml_masks → origin='wsi') -----------
        if do_xml:
            # calcola area utile per distribuire quota tra WSI
            areas = []
            for e in xmls:
                rid = e["record_id"]; lvl = int(e["level"])
                t_npz = out_masks/rid/f"tumor_L{lvl}.npz"
                n_npz = out_masks/rid/f"not_tumor_L{lvl}.npz"
                T,_,_,_ = load_npz_mask(t_npz)
                N,_,_,_ = load_npz_mask(n_npz)
                if T is None or N is None:
                    areas.append(0)
                else:
                    areas.append(max(1, int(T.sum()) + int(N.sum())))
                del T; del N; gc.collect()
            sumA = float(sum(areas)) if sum(areas)>0 else 1.0

            for e, a in zip(xmls, areas):
                if not do_xml: break
                if a <= 0: continue
                rid = e["record_id"]; lvl = int(e["level"])
                wsi_left = max(0, per_wsi_cap - cnt_wsi[rid])
                if wsi_left <= 0: continue

                # quota rispetto alla CAP **per sorgente wsi**
                patient_left_wsi = capP - cnt_patient_src[(subset,pid,"wsi")]
                if patient_left_wsi <= 0:
                    do_xml = False; break
                quota_wsi = int(min(wsi_left, max(1, round(patient_left_wsi * (a/sumA)))))

                need_t = quota_wsi // 2
                need_n = quota_wsi - need_t

                # path maschere
                t_npz = out_masks/rid/f"tumor_L{lvl}.npz"
                n_npz = out_masks/rid/f"not_tumor_L{lvl}.npz"

                # Tumor
                T,_,_,_ = load_npz_mask(t_npz)
                if T is not None and need_t > 0:
                    H, W = T.shape
                    factor_t = 1
                    if H*W > max_mask_pixels:
                        T, factor_t = downscale_mask_binary_power2(T, max_mask_pixels)
                    patch_eff_t = max(1, patch // factor_t)
                    bbT = bbox_from_mask(T)
                    if bbT:
                        picks_t = sample_center_in_mask(T, bbT, patch_eff_t, need_t, rng, max_trials)
                        for x_ds, y_ds in picks_t:
                            x = int(x_ds * factor_t); y = int(y_ds * factor_t)
                            obj = {
                                "key": f"{pid}/{rid}/{x}_{y}",
                                "patient_id": pid,
                                "class_label": e["class_label"],
                                "parent_tumor_subtype": None,
                                "record_id": rid,
                                "subset": subset,
                                "source_rel_path": e.get("wsi_rel_path",""),
                                "source_abs_path": str(Path(cfg["paths"]["wsi_root"]) / e.get("wsi_rel_path","")) if e.get("wsi_rel_path") else "",
                                "origin":"wsi",
                                "coords":{"x":x,"y":y,"level":lvl,"patch_size":int(patch),"downsample_at_level0":None},
                                "roi_coverage":{"tumor":1.0,"not_tumor":0.0},
                                "roi_source":"xml_npz_center"
                            }
                            out_write(subset, obj)
                            cnt_patient[(subset,pid)] += 1
                            cnt_patient_src[(subset,pid,"wsi")] += 1
                            cnt_wsi[rid] += 1
                            patient_left_wsi -= 1
                            if patient_left_wsi <= 0: break
                del T; gc.collect()
                if (capP - cnt_patient_src[(subset,pid,"wsi")]) <= 0:
                    do_xml = False

                if not do_xml: continue  # cap wsi raggiunto

                # Not-tumor
                N,_,_,_ = load_npz_mask(n_npz)
                if N is not None and need_n > 0 and do_xml:
                    H, W = N.shape
                    factor_n = 1
                    if H*W > max_mask_pixels:
                        N, factor_n = downscale_mask_binary_power2(N, max_mask_pixels)
                    patch_eff_n = max(1, patch // factor_n)
                    bbN = bbox_from_mask(N)
                    if bbN:
                        patient_left_wsi = capP - cnt_patient_src[(subset,pid,"wsi")]
                        picks_n = sample_center_in_mask(N, bbN, patch_eff_n, min(need_n, max(0, patient_left_wsi)), rng, max_trials)
                        for x_ds, y_ds in picks_n:
                            x = int(x_ds * factor_n); y = int(y_ds * factor_n)
                            obj = {
                                "key": f"{pid}/{rid}/{x}_{y}",
                                "patient_id": pid,
                                "class_label": "NOT_TUMOR",
                                "parent_tumor_subtype": e["class_label"],
                                "record_id": rid,
                                "subset": subset,
                                "source_rel_path": e.get("wsi_rel_path",""),
                                "source_abs_path": str(Path(cfg["paths"]["wsi_root"]) / e.get("wsi_rel_path","")) if e.get("wsi_rel_path") else "",
                                "origin":"wsi",
                                "coords":{"x":x,"y":y,"level":lvl,"patch_size":int(patch),"downsample_at_level0":None},
                                "roi_coverage":{"tumor":0.0,"not_tumor":1.0},
                                "roi_source":"xml_npz_center"
                            }
                            out_write(subset, obj)
                            cnt_patient[(subset,pid)] += 1
                            cnt_patient_src[(subset,pid,"wsi")] += 1
                            cnt_wsi[rid] += 1
                            if (capP - cnt_patient_src[(subset,pid,"wsi")]) <= 0:
                                do_xml = False; break
                del N; gc.collect()

        # ----------- CHROMO / ONCO (roi_svs → origin='roi') -----------
        if do_roi and (capP - cnt_patient_src[(subset,pid,"roi")]) > 0:
            # cap per-paziente su ROI: impedisce monopolio
            per_patient_cap_roi = min(per_roi_cap, capP)
            patient_left_roi = min(per_patient_cap_roi - cnt_patient_src[(subset,pid,"roi")],
                                   capP - cnt_patient_src[(subset,pid,"roi")])

            if args.debug:
                print(f"[ROI] {pid}: target_per_patient={per_patient_cap_roi}, "
                      f"already={cnt_patient_src[(subset,pid,'roi')]}, left={patient_left_roi}")

            if patient_left_roi > 0:
                # shuffle deterministico delle voci ROI del paziente
                rois_shuffled = list(rois)
                if rois_shuffled:
                    rois_shuffled = list(np.array(rois_shuffled)[rng.permutation(len(rois_shuffled))])
                for e in rois_shuffled:
                    if patient_left_roi <= 0:
                        break
                    rid = e["record_id"]; cls = e["class_label"]
                    roi_files = list(e.get("roi_files", []))
                    # shuffle deterministico dei file ROI
                    if roi_files:
                        roi_files = list(np.array(roi_files)[rng.permutation(len(roi_files))])

                    for rp in roi_files:
                        if patient_left_roi <= 0:
                            break

                        # risolvi path (tiene conto delle radici CHROMO/ONCO)
                        rp_res = resolve_roi_path(rp, cls, cfg)
                        if not rp_res:
                            if args.debug:
                                print(f"[ROI] {pid} {cls}: missing file -> {rp}")
                            continue

                        key = (rid, rp_res)
                        roi_left = max(0, per_roi_cap - cnt_roi[key])
                        if roi_left <= 0:
                            continue

                        # ottieni dimensioni via PIL o OpenSlide
                        size = get_image_size_any(rp_res)
                        if size is None:
                            if args.debug:
                                print(f"[ROI] {pid} {cls}: cannot open -> {rp_res}")
                            continue
                        W, H = size

                        need = min(roi_left, patient_left_roi)
                        if W < patch or H < patch or need <= 0:
                            continue

                        taken = 0; trials = 0
                        max_try = max(max_trials, need*10)
                        while taken < need and trials < max_try:
                            x = int(rng.integers(0, W - patch + 1))
                            y = int(rng.integers(0, H - patch + 1))
                            obj = {
                                "key": f"{pid}/{Path(rp_res).stem}/{x}_{y}",
                                "patient_id": pid,
                                "class_label": cls,
                                "parent_tumor_subtype": None,
                                "record_id": rid,
                                "subset": subset,
                                "source_rel_path": rp_res,
                                "source_abs_path": rp_res,
                                "origin":"roi",
                                "coords":{"x":x,"y":y,"level":0,"patch_size":int(patch),"downsample_at_level0":1.0},
                                "roi_coverage":{"tumor":1.0,"not_tumor":0.0},
                                "roi_source":"svs_roi"
                            }
                            out_write(subset, obj)
                            cnt_patient[(subset,pid)] += 1
                            cnt_patient_src[(subset,pid,"roi")] += 1
                            cnt_roi[key] += 1
                            taken += 1; trials += 1
                            patient_left_roi -= 1
                            if patient_left_roi <= 0:
                                break

    for f in fouts.values(): f.flush(); f.close()
    print(f"[OK] candidates appended: {written_total}")

if __name__ == "__main__":
    main()
