#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass(frozen=True)
class Concept:
    id: int
    name: str
    short_name: str
    group: str
    primary_class: Optional[str]
    prompts: List[str]

    @property
    def prompt(self) -> str:
        # Backward compatibility: single-prompt codepaths
        return self.prompts[0]


def _clean_prompt(s: str) -> str:
    # YAML block scalars may contain newlines; collapse to single line.
    return " ".join(str(s).split())


def _scan_for_bad_unicode(text: str, path: Path) -> None:
    # U+FFFC commonly appears via copy/paste and can break YAML indentation semantics.
    bad = "\uFFFC"
    if bad in text:
        # Provide approximate location(s) for fast debugging.
        lines = text.splitlines()
        hits = []
        for i, ln in enumerate(lines, start=1):
            if bad in ln:
                col = ln.index(bad) + 1
                hits.append(f"line={i},col={col}")
                if len(hits) >= 5:
                    break
        loc = ", ".join(hits) if hits else "unknown"
        raise RuntimeError(
            f"Invalid character U+FFFC found in ontology YAML: {path} ({loc}). "
            "Remove the invisible placeholder character (often from copy/paste) and re-run."
        )


def _parse_prompts(c: Dict[str, Any], *, concept_ref: str) -> List[str]:
    # Support both:
    # - prompt: "..."
    # - prompts: ["...", "..."]
    raw: Union[None, str, List[Any]] = None
    if "prompts" in c and c.get("prompts") is not None:
        raw = c.get("prompts")
    else:
        raw = c.get("prompt")

    prompts: List[str] = []
    if raw is None:
        prompts = []
    elif isinstance(raw, str):
        prompts = [_clean_prompt(raw)]
    elif isinstance(raw, list):
        for x in raw:
            if x is None:
                continue
            prompts.append(_clean_prompt(str(x)))
    else:
        raise RuntimeError(f"Invalid prompts type for {concept_ref}: {type(raw)}")

    prompts = [p for p in prompts if p]
    if not prompts:
        raise RuntimeError(f"Missing prompt(s) for {concept_ref}. Expected 'prompt' or 'prompts'.")
    return prompts


def load_ontology(path: Path) -> Tuple[Dict[str, Any], List[Concept]]:
    txt = path.read_text(encoding="utf-8")
    _scan_for_bad_unicode(txt, path)
    data = yaml.safe_load(txt)
    concepts_raw = data.get("concepts", [])
    if not concepts_raw:
        raise RuntimeError(f"No concepts found in {path}")

    concepts: List[Concept] = []
    seen_short = set()
    seen_id = set()

    for c in concepts_raw:
        cid = int(c.get("id"))
        if cid in seen_id:
            raise RuntimeError(f"Duplicate concept id={cid} in {path}")
        seen_id.add(cid)

        short = str(c.get("short_name", "")).strip()
        if not short:
            raise RuntimeError(f"Missing short_name for concept id={cid} in {path}")
        if short in seen_short:
            raise RuntimeError(f"Duplicate short_name={short} in {path}")
        seen_short.add(short)

        name = str(c.get("name", "")).strip()
        if not name:
            raise RuntimeError(f"Missing name for concept short_name={short} in {path}")

        group = str(c.get("group", "")).strip() or "Other"
        primary = c.get("primary_class", None)
        primary = str(primary).strip() if primary is not None else None
        if primary == "" or primary == "null":
            primary = None

        prompts = _parse_prompts(c, concept_ref=f"concept short_name={short}")

        concepts.append(
            Concept(
                id=cid,
                name=name,
                short_name=short,
                group=group,
                primary_class=primary,
                prompts=prompts,
            )
        )

    concepts = sorted(concepts, key=lambda x: x.id)
    meta = {
        "version": data.get("version", None),
        "name": data.get("name", None),
        "global_instructions": data.get("global_instructions", None),
        "source_path": str(path),
        "n_concepts": len(concepts),
    }
    return meta, concepts


def concepts_to_prompts(concepts: List[Concept]) -> List[str]:
    return [c.prompt for c in concepts]


def concepts_to_prompt_lists(concepts: List[Concept]) -> List[List[str]]:
    return [list(c.prompts) for c in concepts]


def concepts_to_dicts(concepts: List[Concept]) -> List[Dict[str, Any]]:
    return [
        {
            "id": c.id,
            "name": c.name,
            "short_name": c.short_name,
            "group": c.group,
            "primary_class": c.primary_class,
            # Backward compatible single prompt + full list
            "prompt": c.prompt,
            "prompts": list(c.prompts),
        }
        for c in concepts
    ]
