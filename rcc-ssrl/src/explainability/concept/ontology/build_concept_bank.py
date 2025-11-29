#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build concept bank for RCC histology using a VLM.

Input:
- Ontology YAML with RCC concepts.
- CSV of candidate patches with columns:
    image_path, wds_key, class_label
  (produced automatically by build_concept_candidates.py)

- VLM server (e.g. LLaVA-Med) answering concept-level questions in JSON.

Output:
- concepts_rcc_debug.csv with columns:
    concept_name, wds_key, group, class_label

This file is pointed to by concepts.meta_csv in config_concept.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from explainability.concept.ontology.vlm_client import VLMClient  # backend HTTP esistente
from explainability.concept.ontology.vlm_client_hf import VLMClientHF  # nuovo backend HF locale


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
        required=False,
        help="VLM controller URL (e.g. http://localhost:10000) – usato SOLO se backend=http",
    )
    parser.add_argument(
        "--model-name",
        default="Eren-Senoglu/llava-med-v1.5-mistral-7b-hf",
        help="Nome del modello VLM. "
             "Se backend=hf: id Hugging Face (es. Eren-Senoglu/llava-med-v1.5-mistral-7b-hf); "
             "se backend=http: nome registrato sul server (es. llava-med-v1.5-mistral-7b).",
    )
    parser.add_argument(
        "--backend",
        choices=["http", "hf"],
        default="hf",
        help="Tipo di backend VLM: 'hf' = modello locale via Hugging Face (no HTTP), "
             "'http' = controller/worker HTTP (pipeline vecchia).",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for concept bank (concepts_rcc_debug.csv)",
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

    # Scegli il backend in base al flag --backend
    if args.backend == "hf":
        # modello locale via HuggingFace, NESSUN controller HTTP
        vlm = VLMClientHF(
            model_name=args.model_name,
            device=None,        # auto: cuda se disponibile, altrimenti cpu
            dtype="float16",    # va bene con A40
            debug=False,        # o True se vuoi log verbosi
        )
    else:
        # backend http vecchio: richiede --controller
        if not args.controller:
            raise RuntimeError(
                "HTTP backend selected but --controller is None. "
                "Pass --controller http://host:port oppure usa --backend hf."
            )
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
        rng = random.Random(1337)
        rng.shuffle(rows)
        rows = rows[: args.max_images]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_f = open(out_path, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(["concept_name", "wds_key", "group", "class_label"])

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
            try:
                ans = vlm.ask_concept(img, cname, base_prompt)
            except RuntimeError as e:
                # Tipicamente: error_code != 0 dal worker (es. problemi interni llava).
                total_queries += 1
                dt = time.time() - t0

                if vlm.debug:
                    print(
                        f"[BANK DEBUG] RuntimeError for key={key}, concept={cname}, "
                        f"class={cls}, dt={dt:.2f}s\n{e}\n{'-'*80}"
                    )

                # Log minimale anche fuori da debug per i primi casi
                if total_queries <= 10:
                    print(
                        f"[WARN] VLM error for key={key}, concept={cname}: {e}"
                    )
                continue

            dt = time.time() - t0
            total_queries += 1

            # LOG DI DEBUG: prime N risposte parse-ate, anche se poi vengono scartate
            if vlm.debug and ans is not None and total_queries <= 20:
                print(
                    f"[BANK DEBUG] key={key}, concept={cname}, "
                    f"class={cls}, dt={dt:.2f}s\n"
                    f"{json.dumps(ans, indent=2)}\n"
                    f"{'-'*80}"
                )

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
                    print(
                        f"[DEBUG] Empty/invalid VLM answer for key={key}, "
                        f"concept={cname}"
                    )
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
            "Cause probabili: modello VLM non raggiungibile / non caricato, "
            "risposte non parse-abili come JSON, oppure tutte le decisioni present=False "
            f"(backend={args.backend}, model_name={args.model_name})."
        )


if __name__ == "__main__":
    main()
