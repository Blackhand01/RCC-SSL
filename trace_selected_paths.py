#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace selected directories and files into a single review-friendly report.

Layout per file:
    {relative_path} codice <<
    <file content>
    >>

Inspired by a previous Bash workflow, but implemented in pure Python.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Set, Tuple

# -------------------------- Defaults -----------------------------------------

DEFAULT_TARGETS: Tuple[str, ...] = (
    "/home/mla_group_01/rcc-ssrl/src/training/configs/ablations/dino_v3",
    "/home/mla_group_01/rcc-ssrl/src/training/configs/ablations/i_jepa",
    "/home/mla_group_01/rcc-ssrl/src/training/configs/ablations/moco_v3",
    "/home/mla_group_01/rcc-ssrl/src/training/configs/exp_debug_dino_v3_vit.yaml",
    "/home/mla_group_01/rcc-ssrl/src/training/configs/exp_debug_i_jepa_vit.yaml",
    "/home/mla_group_01/rcc-ssrl/src/training/configs/exp_debug_moco_v3_vit.yaml",
    "/home/mla_group_01/rcc-ssrl/src/training/slurm/train_single_node.sbatch",
    "/home/mla_group_01/rcc-ssrl/src/training/scripts/run_ssl_ablations.py",
)

DEFAULT_EXCLUDE_DIRS: Set[str] = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv",
    "logs", "docs",
}

DEFAULT_EXCLUDE_GLOBS: Tuple[str, ...] = (
    "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.o", "*.a",
    "*.err", "*.out", "*.log",
)

DEFAULT_TEXT_EXTS: Set[str] = {
    ".py", ".sh", ".sbatch", ".yaml", ".yml", ".md", ".txt", ".cfg", ".ini", ".json"
}

# -------------------------- Helpers ------------------------------------------

def _is_under_excluded_dir(path: Path, root: Path, excluded: Set[str]) -> bool:
    """Return True if any path component under root is an excluded directory name."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        # path not under root, check all parts anyway
        rel = path
    return any(part in excluded for part in rel.parts)

def _match_any_glob(path: Path, patterns: Iterable[str]) -> bool:
    return any(path.match(pat) for pat in patterns)

def _iter_files_in_dir(d: Path) -> Iterator[Path]:
    for p in d.rglob("*"):
        if p.is_file():
            yield p

def _is_text_candidate(path: Path, text_exts: Set[str]) -> bool:
    if path.suffix.lower() in text_exts:
        return True
    # Fallback: try to treat as text if it is small and decodes cleanly.
    try:
        if path.stat().st_size <= 2 * 1024 * 1024:  # <= 2 MB heuristic
            with path.open("rb") as f:
                sample = f.read(4096)
            sample.decode("utf-8")
            return True
    except Exception:
        pass
    return False

def _safe_read(path: Path, max_bytes: int) -> str:
    """Read at most max_bytes from file and decode as UTF-8 (replace errors)."""
    with path.open("rb") as f:
        data = f.read(max_bytes)
    text = data.decode("utf-8", errors="replace")
    try:
        size = path.stat().st_size
    except Exception:
        size = len(data)
    if size > max_bytes:
        text += "\n[... TRUNCATED: file larger than max_bytes ...]\n"
    return text

def _common_root(paths: List[Path]) -> Path:
    """Compute a sensible common root for relative paths."""
    if not paths:
        return Path("/")
    try:
        common = os.path.commonpath([str(p) for p in paths])
        return Path(common)
    except Exception:
        return Path("/")

def _should_include_file(path: Path, root: Path, *, exclude_dirs: Set[str], exclude_globs: Tuple[str, ...], text_exts: Set[str]) -> bool:
    if _is_under_excluded_dir(path.parent, root, exclude_dirs):
        return False
    if _match_any_glob(path, exclude_globs):
        return False
    return _is_text_candidate(path, text_exts)

# -------------------------- Core ---------------------------------------------

def gather_files(
    targets: Iterable[Path],
    *,
    exclude_dirs: Set[str],
    exclude_globs: Tuple[str, ...],
    text_exts: Set[str],
    root_for_rel: Path | None = None,
) -> List[Path]:
    """
    Expand directories and keep files that look textual,
    applying directory/glob exclusions.
    """
    files: List[Path] = []
    expanded: List[Path] = []

    for t in targets:
        if t.is_dir():
            expanded.extend(_iter_files_in_dir(t))
        elif t.is_file():
            expanded.append(t)  # explicit file always considered; filter later
        else:
            # Silently skip missing targets; the caller can log if desired.
            continue

    if root_for_rel is None and expanded:
        root_for_rel = _common_root(expanded)
    root_for_rel = root_for_rel or Path("/")

    for f in expanded:
        # Keep explicit single files even if they fail the "text" heuristic:
        # Only skip if they match an exclusion or live under excluded dir.
        if f.is_file():
            explicit = f in targets
            if explicit:
                if not _is_under_excluded_dir(f.parent, root_for_rel, exclude_dirs) and not _match_any_glob(f, exclude_globs):
                    files.append(f)
                continue

            # For files discovered via directory expansion, apply full filter:
            if _should_include_file(f, root_for_rel, exclude_dirs=exclude_dirs, exclude_globs=exclude_globs, text_exts=text_exts):
                files.append(f)

    # Deduplicate and sort by relative path for deterministic output
    files = sorted(set(files), key=lambda p: str(p))
    return files

def write_trace_md(files: List[Path], out_path: Path, *, root_for_rel: Path, max_bytes: int) -> None:
    """
    Write the review trace.
    Format per file:
        {relative_path} codice <<
        <content>
        >>
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for p in files:
            try:
                rel = p.relative_to(root_for_rel)
            except Exception:
                rel = p
            out.write(f"{rel} codice <<\n")
            try:
                content = _safe_read(p, max_bytes=max_bytes)
            except Exception as e:
                content = f"[ERROR] Could not read file: {e}\n"
            out.write(content)
            out.write(">>\n\n")

# -------------------------- CLI ----------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Trace selected directories/files into a single Markdown report."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("trace.md"),
        help="Output Markdown file (default: ./trace.md)",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for relative paths (default: auto common root).",
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=2_000_000,
        help="Maximum bytes to read per file (default: 2,000,000).",
    )
    ap.add_argument(
        "--include-ext",
        nargs="*",
        default=sorted(DEFAULT_TEXT_EXTS),
        help="Whitelist of file extensions considered 'textual' for directory walk.",
    )
    ap.add_argument(
        "--exclude-dir",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDE_DIRS),
        help="Directory names to exclude during traversal.",
    )
    ap.add_argument(
        "--exclude-glob",
        nargs="*",
        default=list(DEFAULT_EXCLUDE_GLOBS),
        help="Glob patterns to exclude files (e.g., *.pyc *.out).",
    )
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path(p) for p in DEFAULT_TARGETS],
        help="Explicit target files/directories. Defaults to a preset list.",
    )
    return ap.parse_args(argv)

def main(argv: List[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)

    # Normalize sets/tuples
    text_exts = {e if e.startswith(".") else f".{e}" for e in ns.include_ext}
    exclude_dirs = set(ns.exclude_dir)
    exclude_globs = tuple(ns.exclude_globs) if hasattr(ns, "exclude_globs") else tuple(ns.exclude_glob)

    targets = [Path(p).resolve() for p in ns.paths]
    files = gather_files(
        targets,
        exclude_dirs=exclude_dirs,
        exclude_globs=exclude_globs,
        text_exts=text_exts,
        root_for_rel=ns.root,
    )

    if not files:
        print("[WARN] No files found to trace.", file=sys.stderr)

    root_for_rel = ns.root if ns.root is not None else _common_root(files)
    write_trace_md(files, ns.out, root_for_rel=root_for_rel, max_bytes=ns.max_bytes)

    print(f"[OK] Wrote {ns.out} with {len(files)} files.")
    print(f"[INFO] Root for relative paths: {root_for_rel}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
