#!/usr/bin/env python3
import argparse, json, yaml, random, sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

CLASSES = ["ccRCC","pRCC","CHROMO","ONCO","NOT_TUMOR"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def load_jsonl(p: Path) -> pd.DataFrame:
    rows=[]
    if not Path(p).exists():
        print(f"[WARN] Missing candidates file: {p}", file=sys.stderr)
        return pd.DataFrame()
    with Path(p).open() as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows)

def write_jsonl(df: pd.DataFrame, dst: Path):
    if df is None or len(df)==0:
        dst.write_text("")
        return
    dst.write_text("\n".join(df.apply(lambda r: json.dumps(r.to_dict()), axis=1)) + "\n")

def sample_without_replacement(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n: return df
    return df.sample(n, replace=False, random_state=1337)

def per_class_targets(df: pd.DataFrame, cfg: dict) -> dict:
    caps = {c: int((df["class_label"]==c).sum()) for c in CLASSES}
    max_per_class = (cfg.get("balance", {}) or {}).get("max_target_per_class", None)
    if max_per_class is not None:
        m = int(max_per_class)
        caps = {c: min(v, m) for c,v in caps.items()}
    nonzero = [caps[c] for c in CLASSES if caps[c] > 0]
    target = min(nonzero) if nonzero else 0
    return {c: int(min(target, caps[c])) for c in CLASSES}

def allocate_per_patient(df_c: pd.DataFrame, N_target: int, cap_factor: float) -> pd.DataFrame:
    if df_c is None or len(df_c)==0 or N_target<=0:
        return pd.DataFrame(columns=df_c.columns if df_c is not None else [])

    pats = df_c["patient_id"].unique().tolist()
    rnd = random.Random(1337)
    rnd.shuffle(pats)
    P = len(pats)
    if P==0:
        return pd.DataFrame(columns=df_c.columns)

    q_base   = N_target // P
    residual = N_target - q_base*P
    q_cap    = int(np.ceil(cap_factor*q_base)) if q_base>0 else max(1, int(np.ceil(cap_factor)))

    selected_chunks = []
    taken_idx = set()

    # quota base
    for p in pats:
        df_p = df_c[df_c["patient_id"]==p]
        take = min(q_base, len(df_p))
        if take > 0:
            pick = sample_without_replacement(df_p, take)
            selected_chunks.append(pick)
            taken_idx.update(pick.index.tolist())

    # residual
    if residual > 0:
        pats_sorted = sorted(
            pats,
            key=lambda x: len(df_c[(df_c["patient_id"]==x) & (~df_c.index.isin(taken_idx))]),
            reverse=True
        )
        for p in pats_sorted:
            if residual <= 0: break
            df_p_all = df_c[df_c["patient_id"]==p]
            df_p_left = df_p_all[~df_p_all.index.isin(taken_idx)]
            already = sum(len(ch) for ch in selected_chunks if len(ch)>0 and ch.iloc[0]["patient_id"]==p)
            room    = max(0, q_cap - already)
            can_take = min(room, len(df_p_left))
            if can_take > 0:
                more = sample_without_replacement(df_p_left, can_take)
                if len(more) > 0:
                    if can_take > residual:
                        more = more.iloc[:residual]
                        can_take = residual
                    selected_chunks.append(more)
                    taken_idx.update(more.index.tolist())
                    residual -= can_take

    if selected_chunks:
        return pd.concat(selected_chunks, ignore_index=True)
    return pd.DataFrame(columns=df_c.columns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    random.seed(1337)
    np.random.seed(1337)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = ensure_dir(Path(cfg["paths"]["out_balanced"]))

    paths_cand = Path(cfg["paths"]["out_candidates"])
    D = {s: load_jsonl(paths_cand/f"{s}.jsonl") for s in ("train","val","test")}

    # ===== train: balancing =====
    train = D.get("train", pd.DataFrame()).copy()
    cap_factor = float(cfg.get("balance", {}).get("per_patient_cap_factor", 1.5))

    if len(train)==0:
        print("[WARN] train.jsonl empty — not selecting anything")
        selected_train = pd.DataFrame()
        targets = {c:0 for c in CLASSES}
    else:
        targets = per_class_targets(train, cfg)
        print("[INFO] targets per class (train):", targets)

        sel_parts=[]
        for c in ["ccRCC","pRCC","CHROMO","ONCO"]:
            dfc = train[train["class_label"]==c]
            N   = int(targets.get(c,0))
            sel_c = allocate_per_patient(dfc, N, cap_factor)
            if len(sel_c)>0:
                sel_c = sel_c.copy()
                sel_c["selected_reason"]="balanced"
                sel_parts.append(sel_c)
            print(f"[INFO] selected {len(sel_c)} for class {c} (target={N}, avail={len(dfc)})")

        selected_tumors = pd.concat(sel_parts, ignore_index=True) if sel_parts else pd.DataFrame(columns=train.columns)
        tumor_count_by_patient = selected_tumors.groupby("patient_id").size().to_dict()

        # NOT_TUMOR ~ pari al tumor per-paziente (semplice)
        not_df = train[train["class_label"]=="NOT_TUMOR"].copy()
        N_not = int(targets.get("NOT_TUMOR",0))
        rows=[]
        pats_not = not_df["patient_id"].unique().tolist()
        base = max(1, N_not // max(1,len(pats_not))) if len(pats_not)>0 else 0
        for p in pats_not:
            avail_p = not_df[not_df["patient_id"]==p]
            tumor_p = int(tumor_count_by_patient.get(p, 0))
            target_p = base if tumor_p==0 else tumor_p
            target_p = min(target_p, len(avail_p))
            if target_p>0:
                rows.append(sample_without_replacement(avail_p, target_p))
        selected_not = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=train.columns)

        if len(selected_not)>N_not and N_not>0:
            selected_not = sample_without_replacement(selected_not, N_not)

        selected_train = pd.concat([selected_tumors, selected_not], ignore_index=True)
        selected_train = selected_train.sample(frac=1.0, random_state=1337).reset_index(drop=True)

        print("[INFO] final counts (train):", dict(Counter(selected_train["class_label"]).most_common()))

    # ===== val/test: pass-through =====
    selected_val  = D.get("val",  pd.DataFrame())
    selected_test = D.get("test", pd.DataFrame())

    # Save JSONL
    write_jsonl(selected_train, out_dir/"selected_patches_train.jsonl")
    write_jsonl(selected_val,   out_dir/"selected_patches_val.jsonl")
    write_jsonl(selected_test,  out_dir/"selected_patches_test.jsonl")

    # ===== serializable stats =====
    # nested dict: class_label -> { patient_id -> count }
    if len(selected_train):
        df_pt = (
            selected_train
            .groupby(["class_label","patient_id"])
            .size()
            .reset_index(name="count")
        )
        patients_train_nested = {}
        for _, row in df_pt.iterrows():
            c = row["class_label"]
            p = row["patient_id"]
            cnt = int(row["count"])
            patients_train_nested.setdefault(c, {})[p] = cnt
    else:
        patients_train_nested = {}

    stats = {
      "targets": {k:int(v) for k,v in (targets or {}).items()},
      "selected": {
        "train": dict(Counter(selected_train["class_label"]).most_common()) if len(selected_train) else {},
        "val":   dict(Counter(selected_val["class_label"]).most_common())   if len(selected_val)   else {},
        "test":  dict(Counter(selected_test["class_label"]).most_common())  if len(selected_test)  else {},
      },
      "patients_train": patients_train_nested,
    }
    (out_dir/"stats_train.json").write_text(json.dumps(stats, indent=2))
    print("[OK] balanced selection written to:", str(out_dir))

if __name__ == "__main__":
    main()
