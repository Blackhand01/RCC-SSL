#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build concept bank for RCC histology using a VLM.

Input:
- Ontology YAML with 18 RCC concepts.
- CSV of candidate patches with columns:
    image_path, wds_key, class_label
  (produced automatically by build_concept_candidates.py)

- VLM server (e.g. LLaVA-Med) answering concept-level questions in JSON.

Output:
- concepts_rcc_v1.csv with columns:
    concept_name, wds_key, group, class_label

This file is pointed to by concepts.meta_csv in config_concept.yaml.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

from explainability.concept.ontology.vlm_client import VLMClient


def load_ontology(path: str | Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(open(path, "r"))
    return data["concepts"]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build RCC concept bank via VLM.")
    parser.add_argument("--ontology", required=True, help="Ontology YAML path")
    parser.add_argument(
        "--images-csv",
        required=True,
        help="CSV with columns: image_path,wds_key,class_label",
    )
    parser.add_argument(
        "--controller",
        required=True,
        help="VLM controller URL (e.g. http://localhost:10000)",
    )
    parser.add_argument(
        "--model-name",
        default="llava-med-v1.5-mistral-7b",
        help="VLM model name on the server",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for concept bank (concepts_rcc_v1.csv)",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.6,
        help="Minimal confidence to accept concept as present",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If > 0, limit the number of candidate patches processed (debug).",
    )
    args = parser.parse_args(argv)

    concepts = load_ontology(args.ontology)
    vlm = VLMClient(args.controller, args.model_name)

    # Read candidate patches
    rows: List[Dict[str, str]] = []
    with open(args.images_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("image_path") or not r.get("wds_key"):
                continue
            rows.append(r)

    if not rows:
        raise RuntimeError(
            f"Concept bank: no candidate patches in {args.images_csv}. "
            "Stage 0a (build_concept_candidates) probably failed or produced an empty CSV."
        )

    # Debug mode: limit number of patches (e.g. 100) to reduce queries
    if args.max_images and args.max_images > 0:
        # shuffle with fixed seed for reproducibility
        rng = random.Random(1337)
        rng.shuffle(rows)
        rows = rows[: args.max_images]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_f = open(out_path, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(["concept_name", "wds_key", "group", "class_label"])

    import time
    t_start = time.time()
    total_rows = len(rows)
    total_concepts = len(concepts)
    total_planned_queries = total_rows * total_concepts
    print(
        f"[INFO] Concept bank: candidates={total_rows}, concepts={total_concepts}, "
        f"max_queries={total_planned_queries}, presence_threshold={args.presence_threshold}"
    )

    accepted = 0
    total_queries = 0
    log_every = 200  # stampa ogni N query

    for r_idx, r in enumerate(rows):
        img = r["image_path"]
        key = r["wds_key"]
        cls = r.get("class_label", "")

        for c_idx, c in enumerate(concepts):
            cname = c["name"]
            group = c.get("group")
            base_prompt = c["prompt"]

            t0 = time.time()
            ans = vlm.ask_concept(img, cname, base_prompt)
            dt = time.time() - t0

            total_queries += 1

            if total_queries % log_every == 0:
                elapsed = time.time() - t_start
                avg_per_query = elapsed / max(1, total_queries)
                remaining = total_planned_queries - total_queries
                est_remain = remaining * avg_per_query
                print(
                    f"[PROGRESS] queries={total_queries}/{total_planned_queries} "
                    f"({100.0*total_queries/total_planned_queries:.1f}%), "
                    f"elapsed={elapsed/60:.1f} min, "
                    f"avg_per_query={avg_per_query:.2f} s, "
                    f"ETA~{est_remain/60:.1f} min"
                )

            if not ans:
                # debug minimale
                if total_queries <= 10:
                    print(f"[DEBUG] Empty/invalid VLM answer for key={key}, concept={cname}")
                continue

            present = bool(ans.get("present", False))
            try:
                confidence = float(ans.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            if present and confidence >= args.presence_threshold:
                writer.writerow([cname, key, group, cls])
                accepted += 1

    out_f.close()
    total_elapsed = time.time() - t_start
    print(
        f"[SUMMARY] Concept bank: accepted={accepted}, "
        f"queries={total_queries}, "
        f"elapsed={total_elapsed/60:.1f} min"
    )

    if accepted == 0:
        raise RuntimeError(
            "Concept bank is empty (no accepted concept/key pairs). "
            "Possible causes: presence_threshold too high, VLM misconfigured, "
            "or ontology/prompts not matching the dataset."
        )


if __name__ == "__main__":
    main()
