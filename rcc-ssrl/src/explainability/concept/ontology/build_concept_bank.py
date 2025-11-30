#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump grezzo delle risposte VLM per tutte le coppie (patch, concept).

Input:
- Ontology YAML con i concetti (name, group, primary_class, prompt)
- CSV di candidate patches: image_path, wds_key, class_label
- Modello HF locale (LLaVA-Med) via VLMClientHF

Output:
- concepts_rcc_*.csv con colonne:
    concept_name, wds_key, group, class_label, user_question, assistant_answer
  (tutte le risposte del VLM, senza filtri/soglie, con prompt ripulito)
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize_whitespace(s: str) -> str:
    """Collassa spazi / newline multipli in un'unica riga pulita."""
    return " ".join(str(s).split())


# ----------------------------------------------------------------------
# Ontology loading + validation
# ----------------------------------------------------------------------
def load_ontology(path: str | Path) -> List[Dict[str, Any]]:
    """Load ontology YAML e valida i campi minimi per ciascun concept."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ontology YAML not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "concepts" not in data:
        raise ValueError(f"Ontology YAML must contain a 'concepts' list: {path}")

    concepts_raw = data["concepts"]
    if not isinstance(concepts_raw, list) or not concepts_raw:
        raise ValueError(f"'concepts' must be a non-empty list in {path}")

    concepts: List[Dict[str, Any]] = []
    for idx, c in enumerate(concepts_raw):
        if not isinstance(c, dict):
            raise ValueError(f"Concept #{idx} is not a dict in {path}")

        name = str(c.get("name", "")).strip()
        prompt = str(c.get("prompt", "")).strip()

        if not name:
            raise ValueError(f"Concept #{idx} in {path} has empty 'name'")
        if not prompt:
            raise ValueError(f"Concept '{name}' in {path} has empty 'prompt'")

        concepts.append(
            {
                "name": name,
                "prompt": prompt,
                "group": c.get("group"),
                "primary_class": c.get("primary_class"),
            }
        )

    return concepts


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dump raw VLM outputs for RCC concept bank."
    )
    parser.add_argument("--ontology", required=True, help="Ontology YAML path")
    parser.add_argument(
        "--images-csv",
        required=True,
        help="CSV with columns: image_path,wds_key,class_label",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Eren-Senoglu/llava-med-v1.5-mistral-7b-hf",
        help=(
            "HF model id or local path for the VLM "
            "(e.g. 'Eren-Senoglu/llava-med-v1.5-mistral-7b-hf' or a local directory). "
            "If you launch via run_full_xai.sh, this is overridden by VLM_MODEL_PATH."
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
    args = parser.parse_args(argv)

    # Ontology
    concepts = load_ontology(args.ontology)

    # VLM client
    try:
        from explainability.concept.ontology.vlm_client_hf import VLMClientHF
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "HF backend requested but transformers/torch dependencies are missing. "
            "Install them in the same venv used for explainability."
        ) from exc

    vlm = VLMClientHF(args.model_name)

    # ------------------------------------------------------------------
    # Read candidate patches + validation
    # ------------------------------------------------------------------
    images_csv_path = Path(args.images_csv)
    if not images_csv_path.is_file():
        raise FileNotFoundError(f"Images CSV not found: {images_csv_path}")

    rows: List[Dict[str, str]] = []
    with images_csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required_cols = {"image_path", "wds_key", "class_label"}
        missing = required_cols - set(fieldnames)
        if missing:
            raise ValueError(
                f"Images CSV {images_csv_path} is missing required columns: {sorted(missing)}"
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
            f"Concept bank: no valid candidate patches in {images_csv_path}. "
            "Stage 0a (build_concept_candidates) probably failed or produced an empty CSV."
        )

    # Debug mode: limit number of patches (e.g. 100) to reduce queries
    if args.max_images and args.max_images > 0:
        rng = random.Random(1337)
        rng.shuffle(rows)
        rows = rows[: args.max_images]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    total_rows = len(rows)
    total_concepts = len(concepts)
    total_planned_queries = total_rows * total_concepts
    print(
        f"[INFO] Concept bank RAW: candidates={total_rows}, concepts={total_concepts}, "
        f"max_queries={total_planned_queries}, model_name={args.model_name}"
    )

    total_queries = 0
    written_rows = 0
    skipped_empty_answer = 0
    log_every = 200  # stampa ogni N query

    # ------------------------------------------------------------------
    # Write output CSV
    # ------------------------------------------------------------------
    with out_path.open("w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(
            [
                "concept_name",
                "wds_key",
                "group",
                "class_label",
                "user_question",
                "assistant_answer",
            ]
        )

        for r_idx, r in enumerate(rows):
            img_path = r["image_path"]
            key = r["wds_key"]
            patch_class = r.get("class_label", "")

            for c_idx, c in enumerate(concepts):
                concept_name = c["name"]
                concept_group = c.get("group")
                concept_prompt = c["prompt"]

                # user_question costruita in modo deterministico (non prendiamo il prompt echiato dal modello)
                user_question = _normalize_whitespace(
                    f"For this RCC patch, is the concept '{concept_name}' present? "
                    f"Definition: {concept_prompt}"
                )

                t0 = time.time()
                try:
                    ans = vlm.ask_concept(img_path, concept_name, concept_prompt)
                except RuntimeError as e:
                    total_queries += 1
                    dt = time.time() - t0
                    if getattr(vlm, "debug", False):
                        print(
                            f"[BANK DEBUG] RuntimeError for key={key}, concept={concept_name}, "
                            f"class={patch_class}, dt={dt:.2f}s\n{e}\n{'-'*80}"
                        )
                    if total_queries <= 10:
                        print(
                            f"[WARN] VLM error for key={key}, concept={concept_name}: {e}"
                        )
                    continue

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

                # Risposta vuota o non nel formato atteso
                if not ans or not isinstance(ans, dict):
                    if total_queries <= 10:
                        print(
                            f"[DEBUG] Empty/invalid VLM answer (non-dict) for key={key}, "
                            f"concept={concept_name}"
                        )
                    continue

                # VLMClientHF ora restituisce: {"concept", "user_question", "assistant_answer"}
                assistant_answer = _normalize_whitespace(
                    ans.get("assistant_answer", "") or ""
                )
                user_question_full = _normalize_whitespace(
                    ans.get("user_question", user_question) or ""
                )

                if not assistant_answer:
                    skipped_empty_answer += 1
                    if skipped_empty_answer <= 10:
                        print(
                            f"[DEBUG] Skipping empty assistant_answer for key={key}, "
                            f"concept={concept_name}"
                        )
                    continue

                writer.writerow(
                    [
                        concept_name,
                        key,
                        concept_group,
                        patch_class,
                        user_question_full,
                        assistant_answer,
                    ]
                )
                written_rows += 1

    total_elapsed = time.time() - t_start
    print(
        f"[SUMMARY] Concept bank RAW dump: "
        f"candidates={total_rows}, concepts={total_concepts}, "
        f"queries={total_queries}, written_rows={written_rows}, "
        f"skipped_empty_answer={skipped_empty_answer}, "
        f"elapsed={total_elapsed/60:.1f} min"
    )

    if written_rows == 0:
        raise RuntimeError(
            f"No valid rows written to concept bank CSV {out_path}. "
            "Check VLM responses and ontology/prompts."
        )

    print(f"[OK] Concept bank written to {out_path}")


if __name__ == "__main__":
    main()
