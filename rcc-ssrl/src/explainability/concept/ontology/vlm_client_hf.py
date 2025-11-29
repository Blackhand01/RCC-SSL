#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client locale (no HTTP) per LLaVA-Med via Hugging Face.

- Carica il modello da Hugging Face (es. Eren-Senoglu/llava-med-v1.5-mistral-7b-hf).
- Usa AutoProcessor per gestire testo + immagine.
- Espone ask_concept(image_path, concept_name, base_prompt) con stesso contratto di VLMClient HTTP:
  ritorna dict {"concept", "present", "confidence", "rationale"} oppure None.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class VLMClientHF:
    """
    LLaVA-Med locale via Hugging Face (no controller/worker HTTP).
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: str = "float16",
        debug: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if dtype == "bfloat16":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16

        # debug esplicito > env > default False
        if debug is None:
            self.debug = os.getenv("VLM_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

        if self.debug:
            print(f"[VLM-HF DEBUG] Loading model '{self.model_name}' on {self.device} dtype={torch_dtype}")

        # AutoProcessor/AutoModel per LLaVA-HF (llava-med-v1.5-mistral-7b-hf)
        from transformers import AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=model_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()

        # Alcuni checkpoint HF (es. llava-med-v1.5-mistral-7b-hf) non popolano patch_size nel processor:
        # serve per espandere il token <image> in processing_llava.py. Recuperalo dalla vision_config se manca.
        if getattr(self.processor, "patch_size", None) is None:
            vision_cfg = getattr(self.model.config, "vision_config", None)
            patch_size = getattr(vision_cfg, "patch_size", None) if vision_cfg else None
            patch_size = patch_size or getattr(self.processor.image_processor, "patch_size", None)
            if patch_size is None:
                patch_size = 14  # fallback ragionevole per ViT-L/14
                if self.debug:
                    print("[VLM-HF DEBUG] patch_size non trovato nel processor; uso fallback 14")
            self.processor.patch_size = patch_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Estrai un oggetto JSON da una stringa arbitraria (stessa logica del client HTTP).
        """
        text = text.strip()

        # Tentativo diretto
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Cerca blocco {...} più esterno
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Regex fallback
        import re

        json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def _build_prompt(self, concept_name: str, base_prompt: str) -> str:
        """
        Prompt per LLaVA-HF:
        - include un token <image> per la singola immagine
        - stile USER/ASSISTANT
        - risposta SOLO JSON.
        """
        system = (
            "You are a board-certified renal pathologist. "
            "Answer succinctly and return ONLY a JSON with fields: "
            "concept, present (true/false), confidence (0..1), rationale (<=20 words)."
        )

        user = (
            f"{system}\n\n"
            "Analyse the attached histology patch.\n"
            f"Concept: {concept_name}\n"
            f"Question: {base_prompt}\n"
            'Respond ONLY with a single JSON object with keys: '
            '"concept" (string), "present" (true/false), '
            '"confidence" (0..1 float), "rationale" (<=20 words). '
            'Example: {"concept": "Clear cytoplasm (ccRCC)", "present": true, '
            '"confidence": 0.73, "rationale": "Concise reason"}'
        )

        # 1 immagine -> 1 token <image>
        prompt = f"<image>\nUSER: {user}\nASSISTANT:"
        return prompt

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------
    def ask_concept(
        self,
        image_path: str | Path,
        concept_name: str,
        base_prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 128,
    ) -> Optional[Dict[str, Any]]:
        """
        Esegue una singola query (immagine + concept + domanda) al modello HF locale.

        Ritorna:
            dict normalizzato {"concept", "present", "confidence", "rationale"}
            oppure None se il testo non è parse-abile come JSON.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            if self.debug:
                print(f"[VLM-HF DEBUG] Image not found: {image_path}")
            return None

        image = Image.open(image_path).convert("RGB")
        prompt_str = self._build_prompt(concept_name, base_prompt)

        if self.debug:
            print(
                f"[VLM-HF DEBUG] >>> REQUEST\n"
                f"image={image_path}\n"
                f"concept={concept_name}\n"
                f"prompt_preview={prompt_str[:200]}...\n"
                f"{'-'*60}"
            )

        inputs = self.processor(
            text=[prompt_str],
            images=[image],
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
            }
            # Se temperature > 0, abilita sampling
            if temperature and temperature > 0:
                generate_kwargs.update(
                    dict(
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=0.9,
                    )
                )

            output_ids = self.model.generate(**inputs, **generate_kwargs)

        # Decodifica output (completo; l'estrattore JSON filtrerà eventuale rumore)
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if self.debug:
            print(
                f"[VLM-HF DEBUG] <<< RAW COMPLETION for concept={concept_name}, image={image_path}\n"
                f"{text}\n{'='*80}"
            )

        raw = self._extract_json(text)
        if raw is None or not isinstance(raw, dict):
            if self.debug:
                print("[VLM-HF DEBUG] No valid JSON object found in completion.")
            return None

        # Normalizza
        present_val = raw.get("present", False)
        if isinstance(present_val, str):
            present = present_val.strip().lower() in (
                "true",
                "yes",
                "y",
                "present",
                "1",
            )
        else:
            present = bool(present_val)

        try:
            confidence = float(raw.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        result: Dict[str, Any] = {
            "concept": raw.get("concept", concept_name),
            "present": present,
            "confidence": confidence,
            "rationale": raw.get("rationale", ""),
        }

        if self.debug:
            result["raw_text"] = text
            result["raw_prompt"] = prompt_str
            print(f"[VLM-HF DEBUG] >>> PARSED\n{json.dumps(result, indent=2)}\n{'#'*80}")

        return result
