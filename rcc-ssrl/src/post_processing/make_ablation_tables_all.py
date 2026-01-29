#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List

EXP_RE = re.compile(r"^exp_(\d{8})_(\d{6})_(.+)$")

def parse_exp_dir(p: Path):
    m = EXP_RE.match(p.name)
    if not m:
        return None
    d, t, model = m.group(1), m.group(2), m.group(3)
    return (d, t, model)

def pick_latest_per_model(exp_dirs: List[Path]) -> Dict[str, Path]:
    best: Dict[str, Tuple[Tuple[str, str], Path]] = {}
    for p in exp_dirs:
        parsed = parse_exp_dir(p)
        if parsed is None:
            continue
        d, t, model = parsed
        key = (d, t)
        if model not in best or key > best[model][0]:
            best[model] = (key, p)
    return {m: best[m][1] for m in best}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_root", type=str, required=True,
                    help="Root containing exp_YYYYMMDD_HHMMSS_<model> folders.")
    ap.add_argument("--paper_tables_dir", type=str, required=True,
                    help="Where to write generated .tex tables (e.g. docs/papers/<paper>/tables/generated).")
    ap.add_argument("--run_glob", type=str, default="*abl*",
                    help="Run folder glob inside each experiment (default: *abl*).")
    ap.add_argument("--include", type=str, default=r"^(model|train|optimizer|scheduler|data|dataset|augment|loss|eval)\.",
                    help="Regex include (keys).")
    ap.add_argument("--exclude", type=str, default=r"^train\.num_workers$,^data\.cache",
                    help="Comma-separated regex exclude list.")
    ap.add_argument("--master_tex", type=str, default="ablation_all.tex",
                    help="Master tex filename created inside paper_tables_dir.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]  # .../rcc-ssrl
    gen_script = repo_root / "src" / "post_processing" / "make_ablation_latex_table.py"
    if not gen_script.exists():
        raise FileNotFoundError(f"Missing generator script: {gen_script}")

    experiments_root = Path(args.experiments_root).resolve()
    out_dir = Path(args.paper_tables_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_dirs = [p for p in experiments_root.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    chosen = pick_latest_per_model(exp_dirs)

    if not chosen:
        raise RuntimeError(f"No exp_* directories found under: {experiments_root}")

    produced = []
    for model, exp_dir in sorted(chosen.items(), key=lambda kv: kv[0]):
        out_tex = out_dir / f"ablation_{model}.tex"
        cmd = [
            "python3", str(gen_script),
            "--experiment_dir", str(exp_dir),
            "--run_glob", args.run_glob,
            "--out_tex", str(out_tex),
            "--model_title", model,
            "--include", args.include,
            "--exclude", args.exclude,
        ]
        print(f"[GEN] {model} -> {out_tex}  (from {exp_dir.name})")
        subprocess.run(cmd, check=True)
        produced.append(out_tex)

    # master file: input all generated tables (relative paths)
    master = out_dir / args.master_tex
    rel_inputs = [p.name for p in produced]  # same dir
    lines = []
    lines.append("% =============================================================")
    lines.append("% AUTO-GENERATED MASTER FILE. DO NOT EDIT BY HAND.")
    lines.append("% =============================================================")
    lines.append("% Requires: \\usepackage{booktabs}")
    lines.append("")
    for fn in rel_inputs:
        lines.append(f"\\input{{tables/generated/{fn}}}")
        lines.append("\\par\\medskip")
    master.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Master written: {master}")

if __name__ == "__main__":
    main()
