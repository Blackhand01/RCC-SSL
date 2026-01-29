#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Optional dependency (recommended)
try:
    import yaml  # pip install pyyaml
except Exception as e:
    yaml = None


DEFAULT_EXCLUDE = [
    r"^hydra\.", r"^wandb\.", r"^logging\.", r"^debug\.",
    r"^paths?\.", r"^output_?dir$", r"^run_?dir$", r"^seed$",
    r"^timestamp$", r"^time$", r"^git\.", r"^slurm\.", r"^resume$",
]

# Se vuoi una tabella "paper-grade", tipicamente ti interessano queste aree:
CORE_HINTS = [
    r"^model\.", r"^backbone\.", r"^encoder\.",
    r"^train\.", r"^optimizer\.", r"^sched(uler)?\.",
    r"^data\.", r"^dataset\.", r"^augment\.", r"^loss\.",
    r"^eval\.", r"^calib(ration)?\.", r"^selective\.",
]


def read_structured_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in [".json"]:
        return json.loads(path.read_text())
    if path.suffix.lower() in [".yml", ".yaml"]:
        if yaml is None:
            raise RuntimeError(
                "PyYAML non disponibile. Installa con: pip install pyyaml "
                "oppure usa snapshot .json."
            )
        return yaml.safe_load(path.read_text())
    raise ValueError(f"Formato non supportato: {path}")


def find_config_snapshot(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "experiment_snapshot.yaml",
        run_dir / "experiment_snapshot.yml",
        run_dir / ".hydra" / "config.yaml",
        run_dir / "config.yaml",
        run_dir / "cfg.yaml",
        run_dir / "hparams.yaml",
        run_dir / "params.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(d, list):
        # Lista: serializza compatto (se vuoi indicizzare, cambia qui)
        out[prefix] = d
    else:
        out[prefix] = d
    return out


def normalize_value(v: Any) -> str:
    # Normalizza per comparazioni stabili e output leggibile
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(v)
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def compile_regex_list(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


def keep_key(k: str, include: List[re.Pattern], exclude: List[re.Pattern]) -> bool:
    if any(p.search(k) for p in exclude):
        return False
    if not include:
        return True
    return any(p.search(k) for p in include)


def latex_escape(s: str) -> str:
    # escape minimo LaTeX
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def latex_tt_breakable(s: str) -> str:
    """
    Versione escape + inserisce allowbreak su delimitatori comuni per evitare overfull hbox.
    Nota: in \texttt{} gli underscore vanno escapati, ma poi puoi aggiungere allowbreak.
    """
    s = latex_escape(s)
    # permetti break dopo ., /, =, -, :
    s = s.replace(".", r".\allowbreak ")
    s = s.replace("/", r"/\allowbreak ")
    s = s.replace("=", r"=\allowbreak ")
    s = s.replace("-", r"-\allowbreak ")
    s = s.replace(":", r":\allowbreak ")
    return s


def ablation_id_from_run_name(name: str) -> str:
    m = re.search(r"(abl\d+)", name)
    return m.group(1) if m else name


def compute_common_and_deltas(configs: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    configs: run_name -> flat_key -> normalized_value(str)
    Returns:
      common: keys with identical values across all runs
      deltas: run_name -> keys that differ from common (or missing in common)
    """
    run_names = list(configs.keys())
    all_keys = set()
    for rn in run_names:
        all_keys |= set(configs[rn].keys())

    common: Dict[str, str] = {}
    for k in all_keys:
        vals = []
        missing = False
        for rn in run_names:
            if k not in configs[rn]:
                missing = True
                break
            vals.append(configs[rn][k])
        if missing:
            continue
        if all(v == vals[0] for v in vals[1:]):
            common[k] = vals[0]

    deltas: Dict[str, Dict[str, str]] = {}
    for rn in run_names:
        d = {}
        for k, v in configs[rn].items():
            if k not in common or common[k] != v:
                d[k] = v
        deltas[rn] = d

    return common, deltas


def select_core_subset(keys: List[str], hints: List[re.Pattern], max_rows: int) -> List[str]:
    core = [k for k in keys if any(p.search(k) for p in hints)]
    # se core è vuoto, fallback a primi max_rows ordinati
    if not core:
        return sorted(keys)[:max_rows]
    return sorted(core)[:max_rows]


def render_common_table(common: Dict[str, str], model_title: str, max_rows: int = 30) -> str:
    core_hints = compile_regex_list(CORE_HINTS)
    chosen = select_core_subset(list(common.keys()), core_hints, max_rows)

    lines = []
    lines.append(f"% Auto-generated. Model: {model_title}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(rf"\caption{{Common configuration for {latex_escape(model_title)} (subset of core keys).}}")
    lines.append(r"\begin{tabular}{p{0.40\columnwidth} p{0.55\columnwidth}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Parameter} & \textbf{Value} \\")
    lines.append(r"\midrule")
    for k in chosen:
        kk = latex_tt_breakable(k)
        vv = latex_tt_breakable(common[k])
        lines.append(rf"\texttt{{{kk}}} & \texttt{{{vv}}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def render_delta_table(deltas: Dict[str, Dict[str, str]], model_title: str, max_keys_per_run: int = 25) -> str:
    # Ordina run per ablXX numerico se presente
    def sort_key(rn: str):
        aid = ablation_id_from_run_name(rn)
        m = re.search(r"abl(\d+)", aid)
        return int(m.group(1)) if m else 10**9

    run_names = sorted(deltas.keys(), key=sort_key)

    lines = []
    lines.append(f"% Auto-generated. Model: {model_title}")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(rf"\caption{{Ablation deltas for {latex_escape(model_title)} (only parameters that differ vs. common).}}")
    lines.append(r"\begin{tabular}{p{0.15\textwidth} p{0.80\textwidth}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Ablation} & \textbf{Changed parameters (vs. common)} \\")
    lines.append(r"\midrule")

    for rn in run_names:
        aid = latex_escape(ablation_id_from_run_name(rn))
        diff_items = sorted(deltas[rn].items(), key=lambda kv: kv[0])

        # limita per evitare tabelle infinite
        shown = diff_items[:max_keys_per_run]
        more = len(diff_items) - len(shown)

        cell_lines = []
        if not shown:
            cell_lines.append(r"\textit{(no differences found)}")
        else:
            cell_lines.append(r"\begin{tabular}[t]{@{}l@{}}")
            for k, v in shown:
                kk = latex_tt_breakable(k)
                vv = latex_tt_breakable(v)
                cell_lines.append(rf"\texttt{{{kk}}} = \texttt{{{vv}}} \\")
            if more > 0:
                cell_lines.append(rf"\textit{{(+{more} more keys)}} \\")
            cell_lines.append(r"\end{tabular}")

        cell = "\n".join(cell_lines)
        lines.append(rf"{aid} & {cell} \\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", type=str, required=True,
                    help="Directory containing run subfolders (e.g., exp_*_ablXX).")
    ap.add_argument("--run_glob", type=str, default="*abl*",
                    help="Glob to select run folders (default: *abl*).")
    ap.add_argument("--out_tex", type=str, required=True,
                    help="Output .tex path (will contain common + deltas tables).")
    ap.add_argument("--model_title", type=str, default=None,
                    help="Title used in captions; default = experiment_dir name.")
    ap.add_argument("--include", type=str, default="",
                    help="Comma-separated regex list of keys to include (empty = all).")
    ap.add_argument("--exclude", type=str, default="",
                    help="Comma-separated regex list of keys to exclude (appended to defaults).")
    ap.add_argument("--max_common_rows", type=int, default=30)
    ap.add_argument("--max_delta_keys", type=int, default=25)
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    out_tex = Path(args.out_tex).resolve()
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    model_title = args.model_title or exp_dir.name

    include_patterns = [p.strip() for p in args.include.split(",") if p.strip()]
    exclude_patterns = DEFAULT_EXCLUDE + [p.strip() for p in args.exclude.split(",") if p.strip()]

    include_rx = compile_regex_list(include_patterns)
    exclude_rx = compile_regex_list(exclude_patterns)

    run_dirs = [p for p in exp_dir.glob(args.run_glob) if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under {exp_dir} with glob={args.run_glob}")

    configs: Dict[str, Dict[str, str]] = {}
    missing = []

    for rd in sorted(run_dirs):
        snap = find_config_snapshot(rd)
        if snap is None:
            missing.append(rd.name)
            continue

        cfg = read_structured_file(snap)
        flat = flatten(cfg)

        # filtra
        filtered = {}
        for k, v in flat.items():
            if not k:
                continue
            if keep_key(k, include_rx, exclude_rx):
                filtered[k] = normalize_value(v)

        configs[rd.name] = filtered

    if not configs:
        raise RuntimeError(
            "Nessun run ha uno snapshot config leggibile. "
            "Assicurati di salvare experiment_snapshot.yaml o .hydra/config.yaml per ogni run."
        )

    common, deltas = compute_common_and_deltas(configs)

    tex_parts = []
    tex_parts.append("% =============================================================")
    tex_parts.append("% AUTO-GENERATED FILE. DO NOT EDIT BY HAND.")
    tex_parts.append("% =============================================================")
    tex_parts.append("% Requires: \\usepackage{booktabs}")
    tex_parts.append("% If table* not desired, switch to table in this file.")
    tex_parts.append("")
    tex_parts.append(render_common_table(common, model_title, max_rows=args.max_common_rows))
    tex_parts.append(render_delta_table(deltas, model_title, max_keys_per_run=args.max_delta_keys))

    if missing:
        tex_parts.append("% The following run dirs had NO config snapshot and were skipped:")
        for m in missing:
            tex_parts.append(f"% - {m}")
        tex_parts.append("")

    out_tex.write_text("\n".join(tex_parts), encoding="utf-8")


if __name__ == "__main__":
    main()
