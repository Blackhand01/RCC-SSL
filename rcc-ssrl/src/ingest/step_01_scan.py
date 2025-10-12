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
