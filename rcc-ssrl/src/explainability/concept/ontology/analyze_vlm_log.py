#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# Regex per blocchi REQUEST/RESPONSE
BLOCK_RE = re.compile(
    r"""
    \[VLM-HF\]\s+>>> \s+REQUEST\s*\n
    concept=(?P<concept>.+?)\n
    image=(?P<image>.+?)\n
    question_text[^\n]*\n
    -+\n
    \[VLM-HF\]\s+<<< \s+RESPONSE\s+concept=.*?\n
    raw='(?P<raw>.*?)'\n
    parsed_label=(?P<label>\w+)
    """,
    re.DOTALL | re.VERBOSE,
)

# Regex per DEBUG "Skipping empty assistant_answer"
EMPTY_DEBUG_RE = re.compile(
    r"""
    \[DEBUG\]\s+Skipping\s+empty\s+assistant_answer\s+for\s+key=
    (?P<key>[^,]+),
    \s*concept=(?P<concept>.+)
    """,
    re.VERBOSE,
)


@dataclass
class VLMRecord:
    concept: str
    image: str
    raw: str
    label: str


def parse_log_text(text: str) -> Tuple[List[VLMRecord], Counter]:
    """Parsa il log e restituisce (records, empty_answer_per_concept)."""
    records: List[VLMRecord] = []

    for m in BLOCK_RE.finditer(text):
        concept = m.group("concept").strip()
        image = m.group("image").strip()
        raw = (m.group("raw") or "").strip()
        label = m.group("label").strip()
        records.append(VLMRecord(concept=concept, image=image, raw=raw, label=label))

    empty_counts: Counter = Counter()
    for m in EMPTY_DEBUG_RE.finditer(text):
        concept = m.group("concept").strip()
        empty_counts[concept] += 1

    return records, empty_counts


def compute_stats(records: List[VLMRecord], empty_counts: Counter) -> Dict[str, object]:
    global_labels = Counter()
    concept_stats: Dict[str, Counter] = defaultdict(Counter)

    suspicious_unknown = 0  # Unknown ma testo suggerisce Present/Absent

    for r in records:
        global_labels[r.label] += 1
        c = concept_stats[r.concept]

        c["total"] += 1
        c[f"label_{r.label}"] += 1

        text = r.raw.lower()

        # Pattern di "non vedo l'immagine"
        if "image is not provided" in text or "cannot view the image" in text or "cannot make a decision" in text:
            c["no_image_msg"] += 1

        # Unknown ma con "present"/"absent" nella frase -> probabile parser rotto
        if r.label == "Unknown":
            if "present in the image" in text or "specific concept is present" in text:
                c["unknown_but_sounds_present"] += 1
                suspicious_unknown += 1
            if "absent in the image" in text or "concept is absent" in text:
                c["unknown_but_sounds_absent"] += 1
                suspicious_unknown += 1

    # Includi anche i contatori di risposte vuote
    for concept, n_empty in empty_counts.items():
        concept_stats[concept]["empty_raw_debug"] += n_empty

    # Costruisci output strutturato
    stats = {
        "n_records": len(records),
        "global_label_counts": global_labels,
        "n_concepts": len(concept_stats),
        "suspicious_unknown_count": suspicious_unknown,
        "per_concept": {},
    }

    # Convert Counter in dict serializzabili
    per_concept_serializable: Dict[str, Dict[str, int]] = {}
    for concept, cnt in concept_stats.items():
        per_concept_serializable[concept] = dict(cnt)

    stats["per_concept"] = per_concept_serializable
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analizza log VLM-HF (Stage 0 concept bank) e stampa statistiche."
    )
    parser.add_argument("logfile", type=str, help="Path al file di log da analizzare")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Se >0, mostra solo i top-N concetti per numero di record",
    )
    args = parser.parse_args()

    path = Path(args.logfile)
    if not path.is_file():
        raise SystemExit(f"Log file non trovato: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")

    records, empty_counts = parse_log_text(text)
    stats = compute_stats(records, empty_counts)

    print(f"# Record totali: {stats['n_records']}")
    print(f"# Concetti unici: {stats['n_concepts']}")
    print("# Distribuzione globale delle label:")
    for label, count in stats["global_label_counts"].items():
        print(f"  {label:8s}: {count}")

    print(f"# Unknown sospette (testo suggerisce present/absent): {stats['suspicious_unknown_count']}")

    # Stampa per concetto (ordinato per numero di record)
    per_concept = stats["per_concept"]
    sorted_concepts = sorted(
        per_concept.items(),
        key=lambda kv: kv[1].get("total", 0),
        reverse=True,
    )

    if args.top > 0:
        sorted_concepts = sorted_concepts[: args.top]

    print("\n# Statistiche per concetto:")
    for concept, cnt in sorted_concepts:
        total = cnt.get("total", 0)
        if total == 0:
            continue
        lp = cnt.get("label_Present", 0)
        la = cnt.get("label_Absent", 0)
        lu = cnt.get("label_Unknown", 0)
        no_img = cnt.get("no_image_msg", 0)
        empty_raw = cnt.get("empty_raw_debug", 0)
        unk_pres = cnt.get("unknown_but_sounds_present", 0)
        unk_abs = cnt.get("unknown_but_sounds_absent", 0)

        print(f"- {concept}")
        print(f"    total          : {total}")
        print(f"    Present        : {lp}")
        print(f"    Absent         : {la}")
        print(f"    Unknown        : {lu}")
        print(f"    'no_image' msg : {no_img}")
        print(f"    empty_raw      : {empty_raw}")
        print(f"    Unknown→Present: {unk_pres}")
        print(f"    Unknown→Absent : {unk_abs}")


if __name__ == "__main__":
    main()
