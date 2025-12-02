#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0b: costruzione "concept bank" RCC usando LLaVA-Med v1.5 (HF).

Input:
- Ontology YAML con la lista di concetti (name, group, primary_class, prompt).
- CSV di candidate patches:
    image_path,wds_key,class_label
  generato da build_concept_candidates.py (Stage 0a).

Output:
- concepts_rcc_*.csv con colonne:
    concept_name, wds_key, group, class_label,
    vlm_label, user_question, assistant_answer

DOVE:
- vlm_label in {Present, Absent, Uncertain, Unknown}
- user_question = testo dell'istruzione (senza template interno del modello)
- assistant_answer = testo grezzo generato dal modello

Nota: questo e ancora un dump "raw". Sta a te decidere se filtrare
a posteriori solo i patch con vlm_label == "Present" prima di calcolare
i centroidi concetto -> features.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from explainability.concept.ontology.vlm_client_hf import VLMClientHF


def _normalize_whitespace(s: str) -> str:
    return " ".join(str(s).split())


# ----------------------------------------------------------------------
# Ontology
# ----------------------------------------------------------------------
def load_ontology(path: str | Path) -> List[Dict[str, Any]]:
    """
    Carica l'ontologia RCC (versione 1/2/debug) e valida i campi minimi.

    Richiede per ogni concept:
      - name (string)
      - prompt (string)
    Usa anche facoltativamente:
      - group
      - primary_class
      - short_name
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ontology YAML not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "concepts" not in data:
        raise ValueError(f"Ontology YAML must contain a 'concepts' list: {path}")

    raw_concepts = data["concepts"]
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise ValueError(f"'concepts' must be a non-empty list in {path}")

    out: List[Dict[str, Any]] = []
    for idx, c in enumerate(raw_concepts):
        if not isinstance(c, dict):
            raise ValueError(f"Concept #{idx} is not a dict in {path}")

        name = str(c.get("name", "")).strip()
        prompt = str(c.get("prompt", "")).strip()
        if not name:
            raise ValueError(f"Concept #{idx} in {path} has empty 'name'")
        if not prompt:
            raise ValueError(f"Concept '{name}' in {path} has empty 'prompt'")

        out.append(
            {
                "name": name,
                "prompt": prompt,
                "group": c.get("group"),
                "primary_class": c.get("primary_class"),
                "short_name": c.get("short_name"),
            }
        )

    return out


# ----------------------------------------------------------------------
# CLI + main
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build RCC concept bank via LLaVA-Med v1.5 (HF)."
    )
    parser.add_argument(
        "--ontology",
        required=True,
        help="Ontology YAML path (e.g. ontology_rcc_v2.yaml)",
    )
    parser.add_argument(
        "--images-csv",
        required=True,
        help="CSV with columns: image_path,wds_key,class_label",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        help=(
            "HF model id or local path for the VLM "
            "(default: microsoft/llava-med-v1.5-mistral-7b)."
        ),
    )
    parser.add_argument(
        "--vlm-mode",
        type=str,
        default="concept",
        choices=["concept", "describe"],
        help=(
            "Prompting mode for the VLM. "
            "'concept' = Present/Absent/Uncertain classifier; "
            "'describe' = free-form image description."
        ),
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for concept bank (concepts_rcc_*.csv)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If > 0, limit the number of candidate patches processed (debug).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for subsampling (if max-images > 0).",
    )
    args = parser.parse_args(argv)

    # 1) Ontologia
    concepts = load_ontology(args.ontology)

    # 2) VLM client (HF)
    vlm = VLMClientHF(
        args.model_name,
        mode=args.vlm_mode,
    )

    # 3) Candidate patches: image_path, wds_key, class_label
    images_csv_path = Path(args.images_csv)
    if not images_csv_path.is_file():
        raise FileNotFoundError(f"Images CSV not found: {images_csv_path}")

    rows: List[Dict[str, str]] = []
    with images_csv_path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = {"image_path", "wds_key", "class_label"}
        missing = required - set(fields)
        if missing:
            raise ValueError(
                f"Images CSV {images_csv_path} missing required columns: {sorted(missing)}"
            )

        for r in reader:
            image_path = (r.get("image_path") or "").strip()
            wds_key = (r.get("wds_key") or "").strip()
            class_label = (r.get("class_label") or "").strip()
            if not image_path or not wds_key or not class_label:
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "wds_key": wds_key,
                    "class_label": class_label,
                }
            )

    if not rows:
        raise RuntimeError(
            f"No valid candidate patches in {images_csv_path}. "
            "Stage 0a (build_concept_candidates) probably failed or produced an empty CSV."
        )

    # Subsample opzionale per debug
    if args.max_images and args.max_images > 0:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.max_images]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_patches = len(rows)
    n_concepts = len(concepts)
    if vlm.mode == "describe":
        total_planned = n_patches
    else:
        total_planned = n_patches * n_concepts

    print(
        f"[INFO] Concept bank RAW: candidates={n_patches}, concepts={n_concepts}, "
        f"max_queries={total_planned}, model_name={args.model_name}"
    )

    total_queries = 0
    written_rows = 0
    skipped_empty = 0

    t0_global = time.time()
    log_every = 200  # stampa ogni N query

    with out_path.open("w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(
            [
                "concept_name",
                "wds_key",
                "group",
                "class_label",
                "vlm_label",
                "user_question",
                "assistant_answer",
            ]
        )

        for r in rows:
            img_path = r["image_path"]
            key = r["wds_key"]
            patch_class = r["class_label"]

            if vlm.mode == "describe":
                concepts_iter = [None]
            else:
                concepts_iter = concepts

            for c in concepts_iter:
                if vlm.mode == "describe":
                    concept_name = "image_description"
                    concept_group = None
                    concept_prompt = ""
                else:
                    concept_name = c["name"]
                    concept_group = c.get("group")
                    concept_prompt = c["prompt"]

                t0 = time.time()
                try:
                    if vlm.mode == "describe":
                        ans = vlm.ask_concept(
                            img_path,
                            concept_name,
                            concept_prompt,
                            temperature=0.7,
                            max_new_tokens=256,
                        )
                    else:
                        ans = vlm.ask_concept(
                            img_path,
                            concept_name,
                            concept_prompt,
                        )
                except RuntimeError as e:
                    total_queries += 1
                    dt = time.time() - t0
                    print(
                        f"[WARN] VLM runtime error for key={key}, "
                        f"concept={concept_name}, dt={dt:.2f}s: {e}"
                    )
                    continue

                dt = time.time() - t0
                total_queries += 1

                if total_queries % log_every == 0 or total_queries == 1:
                    elapsed = time.time() - t0_global
                    avg = elapsed / max(1, total_queries)
                    remaining = total_planned - total_queries
                    eta_min = (remaining * avg) / 60.0
                    print(
                        f"[PROGRESS] queries={total_queries}/{total_planned} "
                        f"({100.0*total_queries/total_planned:.1f}%), "
                        f"elapsed={elapsed/60:.1f} min, "
                        f"avg={avg:.2f} s/query, "
                        f"ETA~{eta_min:.1f} min"
                    )

                if not ans or not isinstance(ans, dict):
                    skipped_empty += 1
                    if skipped_empty <= 10:
                        print(
                            f"[DEBUG] Empty/invalid VLM answer for "
                            f"key={key}, concept={concept_name}"
                        )
                    continue

                raw = (ans.get("raw_response") or "").strip()
                question_text = ans.get("question_text") or ""
                vlm_label = ans.get("parsed_label") or "Unknown"

                # In 'describe' mode the parser will usually emit 'Unknown' because
                # we no longer force a single-word categorical answer.
                if not raw:
                    skipped_empty += 1
                    if skipped_empty <= 10:
                        print(
                            f"[DEBUG] Skipping empty assistant_answer for "
                            f"key={key}, concept={concept_name}"
                        )
                    continue

                writer.writerow(
                    [
                        concept_name,
                        key,
                        concept_group,
                        patch_class,
                        vlm_label,
                        _normalize_whitespace(question_text),
                        _normalize_whitespace(raw),
                    ]
                )
                written_rows += 1

    elapsed_total = time.time() - t0_global
    print(
        f"[SUMMARY] Concept bank RAW dump: "
        f"candidates={n_patches}, concepts={n_concepts}, "
        f"queries={total_queries}, written_rows={written_rows}, "
        f"skipped_empty_answer={skipped_empty}, "
        f"elapsed={elapsed_total/60:.1f} min"
    )

    if written_rows == 0:
        raise RuntimeError(
            f"No valid rows written to concept bank CSV {out_path}. "
            "Check VLM responses and ontology/prompts."
        )

    print(f"[OK] Concept bank written to {out_path}")

if __name__ == "__main__":
    main()
