#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple HTTP client for a vision-language model (e.g. LLaVA-Med).

It expects the model to return a JSON string with:
{"concept": "<name>", "present": true/false, "confidence": 0..1, "rationale": "<<=20 words>"}
"""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from PIL import Image


class VLMClient:
    """Minimal client for a controller+worker style VLM server."""

    def __init__(
        self,
        controller_url: str,
        model_name: str,
        timeout: int = 120,
        debug: Optional[bool] = None,
        stop_string: Optional[str] = None,
    ) -> None:
        self.controller_url = controller_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        # llava.serve.model_worker expects a non-None `stop` string; if None is
        # passed, KeywordsStoppingCriteria tokenization raises and the worker
        # returns error_code=1 (server_error_msg). Use a safe default separator.
        self.stop_string = stop_string or "###"

        # debug esplicito > env > default False
        if debug is None:
            self.debug = os.getenv("VLM_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image(pil_img: Image.Image) -> str:
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _build_prompt(self, concept_name: str, base_prompt: str) -> str:
        """
        Costruisce il prompt testuale per LLaVA-Med.

        PUNTO CHIAVE: inseriamo esplicitamente un token <image> in testa,
        così len(images) == prompt.count("<image>") e il worker non alza
        più `ValueError: Number of images does not match number of <image> tokens`.
        """
        system = (
            "You are a board-certified renal pathologist. "
            "Answer succinctly and return a JSON with fields: "
            "concept, present (true/false), confidence (0..1), rationale (<=20 words)."
        )

        user = (
            f"{system}\n\n"
            "Analyse the attached histology patch.\n"
            f"Concept: {concept_name}\n"
            f"Question: {base_prompt}\n"
            'Return ONLY a JSON object with keys: concept (string), present (true/false), '
            'confidence (0..1 float), rationale (<=20 words). '
            'Example: {"concept": "Clear cytoplasm (ccRCC)", "present": true, "confidence": 0.73, '
            '"rationale": "Concise reason"}'
        )

        # Formato compatibile con llava.serve.model_worker:
        # una immagine -> un solo token <image> nel prompt.
        return f"<image>\n{user}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from text, trying multiple strategies.
        Returns the parsed dict or None if no valid JSON found.
        """
        text = text.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find outermost {...}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Use regex to find potential JSON objects
        import re
        json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in reversed(matches):  # Try longest first
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    # ------------------------------------------------------------------
    # Main API
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
        Invia (image, concept, question) al VLM e ritorna un dict Python
        già normalizzato, oppure None in caso di problemi "soft".

        Se il worker risponde con error_code != 0, viene alzato RuntimeError
        (gestito dal chiamante in build_concept_bank).
        """
        image_path = str(image_path)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            img_b64 = self._encode_image(im)

        prompt_str = self._build_prompt(concept_name, base_prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt_str,
            "images": [img_b64],
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
            # llava.serve.model_worker vuole una stringa non-None
            "stop": self.stop_string,
        }

        if self.debug:
            print(
                f"[VLM DEBUG] >>> REQUEST\n"
                f"concept={concept_name}\n"
                f"image={image_path}\n"
                f"payload_keys={list(payload.keys())}\n"
                f"prompt_preview={prompt_str[:200]}...\n"
                f"{'-'*60}"
            )

        # Usa l'endpoint streaming, che ritorna chunk JSON con "text"
        url = f"{self.controller_url}/worker_generate_stream"
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=self.timeout)
            resp.raise_for_status()

            text = ""
            for chunk in resp.iter_lines(delimiter=b"\0"):
                if chunk:
                    try:
                        data = json.loads(chunk.decode())
                        text = data.get("text", "")
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            if self.debug:
                print(
                    f"[VLM DEBUG] HTTP error for concept={concept_name}, "
                    f"image={image_path}: {e}"
                )
            # Errore "hard" di rete -> nessuna risposta utile
            return None

        if self.debug:
            print(
                f"[VLM DEBUG] <<< RAW RESPONSE\n"
                f"status={resp.status_code}\n"
                f"response_type={type(data)}\n"
                f"response_keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}\n"
                f"{'-'*60}"
            )

        # Gestione esplicita di error_code dal worker/controller
        if isinstance(data, dict) and data.get("error_code", 0) != 0:
            msg = data.get("text", "") or data.get("message", "")
            if self.debug:
                print(
                    f"[VLM DEBUG] Worker returned error_code={data.get('error_code')}: {msg}"
                )
            raise RuntimeError(
                f"VLM worker error (code={data.get('error_code')}): {msg}"
            )

        # Extract text from response
        if not isinstance(data, dict) or "text" not in data:
            if self.debug:
                print(
                    f"[VLM DEBUG] Invalid response format for concept={concept_name}, "
                    f"image={image_path}: {data}"
                )
            return None

        full = data["text"].strip()

        if self.debug:
            print(
                f"[VLM DEBUG] <<< RAW COMPLETION for concept={concept_name}, "
                f"image={image_path}\n{full}\n{'='*80}"
            )

        # Alcuni modelli possono aggiungere ```json ... ```: ripulisci
        for token in ("```json", "```JSON", "```"):
            if token in full:
                full = full.replace(token, "")
        full = full.strip()

        # Usa il parser robusto invece di duplicare la logica
        raw = self._extract_json(full)
        if raw is None:
            if self.debug:
                print("[VLM DEBUG] No valid JSON object found in completion.")
            return None

        if not isinstance(raw, dict):
            if self.debug:
                print(f"[VLM DEBUG] Parsed JSON is not a dict: {raw}")
            return None

        # Normalizza tipi e range così build_concept_bank può fidarsi
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

        # In debug aggiungi anche il testo grezzo e il prompt originali
        if self.debug:
            result["raw_text"] = full
            result["raw_prompt"] = prompt_str

            print(
                f"[VLM DEBUG] >>> PARSED\n"
                f"{json.dumps(result, indent=2)}\n"
                f"{'#'*80}"
            )

        return result
