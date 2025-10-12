#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# Configuration (edit if needed)
# -------------------------------------------------------------------
HOME_ROOT="/home/mla_group_01/rcc-ssrl"
SCRATCH_ROOT="/beegfs-scratch/mla_group_01/rcc-ssrl"

# -------------------------------------------------------------------
# 1) Create directory structure
# -------------------------------------------------------------------
mkdir -p "$HOME_ROOT/src" \
         "$HOME_ROOT/configs" \
         "$HOME_ROOT/scripts" \
         "$HOME_ROOT/slurm" \
         "$SCRATCH_ROOT/data/raw" \
         "$SCRATCH_ROOT/data/processed" \
         "$SCRATCH_ROOT/models" \
         "$SCRATCH_ROOT/logs" \
         "$SCRATCH_ROOT/outputs"

# -------------------------------------------------------------------
# 2) scripts/ingest_build_manifest.py
#    - Scans $SCRATCH_ROOT/data/raw according to RCC WSI layout
#    - Parses slide filenames, XML ROI presence, Excel patient mapping
#    - Emits CSV manifest with: slide_id, patient_id, class, has_not_tumour, paths
# -------------------------------------------------------------------
cat > "$HOME_ROOT/scripts/ingest_build_manifest.py" <<'PY'
#!/usr/bin/env python3
"""
Build a manifest CSV from raw RCC WSI dataset.

Heuristics:
- Classes: ['ccRCC','pRCC','CHROMO','ONCO']
- Slide pattern: HPxx... e.g., HP02.10180.1A2.ccRCC.scn; patient_id = first two tokens "HP..." (until first space/extension)
- For CHROMO/ONCO, use Excel correspondence if present to map ID ranges->patient_id
- For ccRCC/pRCC, prefer patient parsed from filename; mark has_not_tumour=True if XML contains non-tumour annotations
Outputs:
- $SCRATCH_ROOT/data/processed/manifest.csv
"""
import os, re, csv, sys, json
from pathlib import Path
try:
    import pandas as pd
except Exception:
    pd = None

SCRATCH_ROOT = os.environ.get("SCRATCH", "/beegfs-scratch/mla_group_01/rcc-ssrl")
RAW = Path(SCRATCH_ROOT) / "data" / "raw"
OUT = Path(SCRATCH_ROOT) / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "manifest.csv"

CLASS_DIRS = {
    "ccRCC": ["ccRCC", "pre/ccRCC"],
    "pRCC": ["pRCC", "pre/pRCC"],
    "CHROMO": ["CHROMO", "Annotations_chromo", "annotations_chromo", "CHROMO/Annotations_chromo"],
    "ONCO": ["ONCOCYTOMA", "Annotations_onco", "annotations_onco", "ONCOCYTOMA/Annotations_onco"],
}

WSI_EXT = {".svs", ".scn", ".tif", ".ndpi"}
XML_KEYS = [
    ("ccRCC", ["ccRCC_xml","pre_ccRCC_xml"]),
    ("pRCC", ["pRCC_xml","pre_pRCC_xml"])
]

def find_files(root: Path, exts):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def parse_patient_from_filename(name: str):
    # Examples: HP02.10180.1A2.ccRCC.scn -> patient "HP02.10180"
    m = re.match(r'(HP[\d\.]+)', name)
    return m.group(1) if m else None

def load_excel_map_if_any(class_name: str):
    # Look for any .xlsx under class folders (CHROMO/ONCO carry patient correspondences)
    excel_rows = []
    for sub in CLASS_DIRS.get(class_name, []):
        base = RAW / sub
        if not base.exists(): continue
        for x in base.rglob("*.xlsx"):
            if pd is None: continue
            try:
                df = pd.read_excel(x)
                df.columns = [str(c).strip() for c in df.columns]
                excel_rows.append((x, df))
            except Exception:
                continue
    return excel_rows  # list of (path, DataFrame)

def patient_from_excel_index(index_val, excel_df):
    # Excel schema example: "ID (slide range)" | "Patient_ID"
    # Accept "1-3" or "8" ; return patient_id string
    for _, row in excel_df.iterrows():
        id_range = str(row.iloc[0]).strip()
        pat = str(row.iloc[1]).strip()
        if '-' in id_range:
            a,b = id_range.split('-',1)
            try:
                a,b = int(a), int(b)
            except: continue
            if isinstance(index_val, int) and a <= index_val <= b:
                return pat
        else:
            try:
                if int(id_range) == int(index_val):
                    return pat
            except:
                continue
    return None

def infer_index_from_filename(fname: str):
    # Try to catch a leading index like CHROMO_001.svs, or 'ID12', or '#5'
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else None

def scan_xml_has_nontumour(xml_path: Path):
    # Heuristic: search for tags/attributes containing 'non' and 'tum' together, or names 'not-tumour', 'normal', 'fiber', 'necrosis'
    try:
        text = xml_path.read_text(errors="ignore")
    except Exception:
        return False
    triggers = ["non_tumor","non-tumor","not_tumour","not-tumour","normal","fiber","necrosis","stroma"]
    t = text.lower()
    return any(k in t for k in triggers)

def xml_index(root: Path):
    idx = {}
    for cls, xml_dirs in XML_KEYS:
        for d in xml_dirs:
            base = root / cls / d if (root/cls/d).exists() else root / "pre" / cls / d
            if not base.exists(): continue
            for x in base.rglob("*.xml"):
                slide_stem = x.stem.replace(".xml","")
                idx.setdefault(slide_stem, []).append(x)
    return idx

def main():
    xml_map = xml_index(RAW)
    rows = []
    for cls in ["ccRCC","pRCC","CHROMO","ONCO"]:
        # Aggregate WSI paths under known dirs
        wsi_paths = []
        for sub in CLASS_DIRS.get(cls, []):
            base = RAW / sub
            if base.exists():
                wsi_paths += list(find_files(base, WSI_EXT))

        # Optional excel mapping (CHROMO/ONCO more likely)
        excel_maps = load_excel_map_if_any(cls)

        for wsi in sorted(set(wsi_paths)):
            fname = wsi.name
            slide_id = fname
            patient_id = parse_patient_from_filename(fname)

            # If patient not parsable, try Excel mapping by numeric index
            if patient_id is None and excel_maps:
                idx_val = infer_index_from_filename(fname)
                for _, df in excel_maps:
                    cand = patient_from_excel_index(idx_val, df)
                    if cand:
                        patient_id = cand; break

            # XML ROI presence and non-tumour flag
            xmls = xml_map.get(Path(fname).stem, [])
            has_not = any(scan_xml_has_nontumour(x) for x in xmls)

            rows.append({
                "slide_id": slide_id,
                "patient_id": patient_id or "",
                "class": cls,
                "wsi_path": str(wsi),
                "xml_paths": "|".join(map(str, xmls)) if xmls else "",
                "has_not_tumour": int(bool(has_not))
            })

    # Write CSV
    fieldnames = ["slide_id","patient_id","class","wsi_path","xml_paths","has_not_tumour"]
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote manifest with {len(rows)} rows at {MANIFEST}")

if __name__ == "__main__":
    main()
PY
chmod +x "$HOME_ROOT/scripts/ingest_build_manifest.py"

# -------------------------------------------------------------------
# 3) scripts/list_patients_per_class.py
#    - Reads manifest.csv and outputs per-class patient lists and counts
# -------------------------------------------------------------------
cat > "$HOME_ROOT/scripts/list_patients_per_class.py" <<'PY'
#!/usr/bin/env python3
"""
Summarize unique patients per class from manifest.csv.
Outputs:
- $SCRATCH_ROOT/outputs/patient_lists/<CLASS>.txt
- Prints a compact table on stdout
"""
import os, sys
from pathlib import Path
import csv
from collections import defaultdict, OrderedDict

SCRATCH_ROOT = os.environ.get("SCRATCH", "/beegfs-scratch/mla_group_01/rcc-ssrl")
MANIFEST = Path(SCRATCH_ROOT) / "data" / "processed" / "manifest.csv"
OUTDIR = Path(SCRATCH_ROOT) / "outputs" / "patient_lists"
OUTDIR.mkdir(parents=True, exist_ok=True)

if not MANIFEST.exists():
    print(f"Manifest not found at {MANIFEST}", file=sys.stderr)
    sys.exit(2)

per_class_patients = defaultdict(set)
with MANIFEST.open() as f:
    r = csv.DictReader(f)
    for row in r:
        cls = row["class"].strip()
        pid = row["patient_id"].strip() or f"UNKNOWN:{row['slide_id']}"
        per_class_patients[cls].add(pid)

order = ["ccRCC","pRCC","CHROMO","ONCO"]
print("# Patients per class")
print("{:10s}  {:>5s}".format("Class","N"))
for cls in order + [c for c in per_class_patients.keys() if c not in order]:
    plist = sorted(per_class_patients[cls])
    with (OUTDIR / f"{cls}.txt").open("w") as g:
        for p in plist:
            g.write(p + "\n")
    print("{:10s}  {:>5d}".format(cls, len(plist)))
print(f"Lists written under {OUTDIR}")
PY
chmod +x "$HOME_ROOT/scripts/list_patients_per_class.py"

# -------------------------------------------------------------------
# 4) slurm/build_manifest.sbatch
# -------------------------------------------------------------------
cat > "$HOME_ROOT/slurm/build_manifest.sbatch" <<'SLURM'
#!/bin/bash
#SBATCH --job-name=build_manifest
#SBATCH --partition=cpu_sapphire
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl
#SBATCH --output=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.out
#SBATCH --error=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.err

module purge
module load python/3.10
source $HOME/.venvs/rcc/bin/activate || true

export PYTHONUNBUFFERED=1
export SCRATCH=/beegfs-scratch/mla_group_01/rcc-ssrl

python $HOME/rcc-ssrl/scripts/ingest_build_manifest.py
SLURM

# -------------------------------------------------------------------
# 5) slurm/list_patients.sbatch
# -------------------------------------------------------------------
cat > "$HOME_ROOT/slurm/list_patients.sbatch" <<'SLURM'
#!/bin/bash
#SBATCH --job-name=list_patients
#SBATCH --partition=cpu_sapphire
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl
#SBATCH --output=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.out
#SBATCH --error=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.err

module purge
module load python/3.10
source $HOME/.venvs/rcc/bin/activate || true

export PYTHONUNBUFFERED=1
export SCRATCH=/beegfs-scratch/mla_group_01/rcc-ssrl

python $HOME/rcc-ssrl/scripts/list_patients_per_class.py
SLURM

echo "Done. Submit with:"
echo "  sbatch $HOME_ROOT/slurm/build_manifest.sbatch"
echo "  sbatch $HOME_ROOT/slurm/list_patients.sbatch"
