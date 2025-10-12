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
