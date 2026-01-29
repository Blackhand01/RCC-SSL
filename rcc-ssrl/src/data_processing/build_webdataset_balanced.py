#!/usr/bin/env python3
import argparse, json, yaml, io, sys, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import webdataset as wds
from tqdm import tqdm

# OpenSlide is optional: if not available, handle only raster images
try:
    import openslide
    HAS_OPENSLIDE = True
except Exception:
    HAS_OPENSLIDE = False

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def read_region_any(slide_path: str, level: int, x: int, y: int, size: int) -> np.ndarray:
    """
    - If openslide is available and file is a supported WSI => use read_region.
    - Otherwise try to open with PIL (ROI raster). In that case level==0 is assumed and cropped.
    """
    # 1) attempt OpenSlide
    if HAS_OPENSLIDE:
        try:
            with openslide.OpenSlide(slide_path) as sl:
                # Check level
                if level < 0 or level >= sl.level_count:
                    # If level is invalid, fallback to nearest (0 generally)
                    level = min(max(level, 0), sl.level_count - 1)
                ds = sl.level_downsamples[level]
                # read_region prende coordinate al livello 0 scalate
                im = sl.read_region((int(round(x * ds)), int(round(y * ds))), level, (size, size)).convert("RGB")
                return np.array(im)
        except Exception:
            # falls through to raster fallback below
            pass

    # 2) raster image fallback (ROI)
    try:
        im = Image.open(slide_path).convert("RGB")
        # Here we assume level==0 and coordinates are already at image resolution
        # protection: if crop goes beyond borders, PIL raises or returns different shape
        w, h = im.size
        if x < 0 or y < 0 or x + size > w or y + size > h:
            # no crop possible
            raise ValueError(f"Crop out of bounds: {(x,y,size)} on image size {(w,h)}")
        return np.array(im.crop((x, y, x + size, y + size)))
    except Exception as e:
        raise RuntimeError(f"Cannot open/crop raster image: {slide_path} ({e})")

def save_sample(writer: wds.ShardWriter, key: str, img_arr: np.ndarray, meta: dict, img_fmt: str = "jpg"):
    buf = io.BytesIO()
    if img_fmt.lower() == "png":
        Image.fromarray(img_arr).save(buf, format="PNG")
        ext = "png"
    else:
        Image.fromarray(img_arr).save(buf, format="JPEG", quality=90)
        ext = "jpg"
    sample = {
        "__key__": key,
        f"img.{ext}": buf.getvalue(),
        "meta.json": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
    }
    writer.write(sample)

def load_selected(p: Path):
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path a config.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Quali split generare (default: train val test)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Se >0, limita il numero di campioni processati per split (debug)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Se presente, non rigenera shard se esistono già (stesso pattern)")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    out_root = Path(cfg["paths"]["out_webdataset"]); ensure_dir(out_root)
    img_fmt = cfg.get("wds", {}).get("image_format", "jpg")
    maxcount = int(cfg.get("wds", {}).get("samples_per_shard", 5000))

    if not HAS_OPENSLIDE:
        print("[WARN] OpenSlide not available: WSIs will be attempted but will likely fail; "
              "proceeding only with raster ROI.",
              file=sys.stderr)

    for subset in args.splits:
        sel_path = Path(cfg["paths"]["out_balanced"]) / f"selected_patches_{subset}.jsonl"
        if not sel_path.exists():
            print(f"[WARN] missing {sel_path}"); continue
        data = load_selected(sel_path)
        if not data:
            print(f"[WARN] empty selection for {subset}"); continue

        dst = ensure_dir(out_root / subset)

        # Optional: skip if shards already exist (heuristic)
        if args.skip_existing:
            existing = list(dst.glob("shard-*.tar"))
            if existing:
                print(f"[SKIP] {subset}: shards already present ({len(existing)} files). Use without --skip-existing to regenerate.")
                continue

        writer = wds.ShardWriter(str(dst / "shard-%06d.tar"), maxcount=maxcount)

        n = 0
        errors = 0
        pbar = tqdm(data if args.limit <= 0 else data[:args.limit], desc=f"{subset}")
        for it in pbar:
            try:
                src = it.get("source_abs_path") or it.get("source_rel_path")
                if not src:
                    raise ValueError("Manca source_abs_path/source_rel_path")
                ps = int(it["coords"]["patch_size"])
                x, y, level = int(it["coords"]["x"]), int(it["coords"]["y"]), int(it["coords"]["level"])

                arr = read_region_any(src, level, x, y, ps)
                # filter: discard patches that don't match the required size
                if arr.shape[0] != ps or arr.shape[1] != ps:
                    continue

                key = it["key"]
                meta = {
                    "class_label": it["class_label"],
                    "parent_tumor_subtype": it.get("parent_tumor_subtype"),
                    "patient_id": it["patient_id"],
                    "subset": subset,
                    "wsi_or_roi": it.get("source_rel_path", ""),
                    "coords": it["coords"],
                    "roi_coverage": it.get("roi_coverage", {}),
                    "origin": it.get("origin", "wsi"),
                    # optional: also keep record_id for traceability
                    "record_id": it.get("record_id", ""),
                }
                save_sample(writer, key, arr, meta, img_fmt=img_fmt)
                n += 1
            except Exception as e:
                errors += 1
                # show only some errors to avoid flooding the log
                if errors <= 20:
                    print(f"[ERR] {subset} key={it.get('key','?')} src={it.get('source_abs_path','?')} -> {e}", file=sys.stderr)
                    # optional: print stacktrace the first times
                    traceback.print_exc(limit=1)
                pbar.set_postfix({"ok": n, "err": errors})

        writer.close()
        print(f"[OK] {subset}: {n} samples (errors={errors}) -> {dst}")

if __name__ == "__main__":
    main()
