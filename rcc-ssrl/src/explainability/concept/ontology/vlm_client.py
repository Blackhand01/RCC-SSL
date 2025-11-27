#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple HTTP client for a vision-language model (e.g. LLaVA-Med).

It expects the model to return a JSON string with:
{"concept": "<name>", "present": true/false, "confidence": 0..1, "rationale": "<<=20 words>"}
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import base64
import requests
from PIL import Image


class VLMClient:
    """Minimal client for a controller+worker style VLM server."""

    def __init__(
        self,
        controller_url: str,
        model_name: str,
        timeout: int = 120,
    ) -> None:
        self.controller_url = controller_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    @staticmethod
    def _encode_image(pil_img: Image.Image) -> str:
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _build_prompt(self, concept_name: str, base_prompt: str) -> str:
        system = (
            "You are a board-certified renal pathologist. "
            "Answer succinctly and return a JSON with fields: "
            "concept, present (true/false), confidence (0..1), rationale (<=20 words)."
        )
        return f"{system}\nConcept: {concept_name}\nQuestion: {base_prompt}\nReturn JSON only."

    def ask_concept(
        self,
        image_path: str | Path,
        concept_name: str,
        base_prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> Optional[Dict[str, Any]]:
        """Send (image, concept, question) to the VLM and parse JSON answer.

        Expectation: the worker at /worker_generate_stream is a LLaVA-style
        endpoint that streams JSON objects separated by NUL ('\0'), each
        with at least a 'text' field and an 'error_code'.
        """
        image_path = str(image_path)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            img_b64 = self._encode_image(im)

        payload = {
            "model": self.model_name,
            "prompt": self._build_prompt(concept_name, base_prompt),
            "images": [img_b64],
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
        }

        url = f"{self.controller_url}/worker_generate_stream"
        session = requests.Session()
        resp = session.post(url, json=payload, stream=True, timeout=self.timeout)
        resp.raise_for_status()

        last_text: Optional[str] = None

        # LLaVA streams JSON chunks terminated by '\0'; each chunk is a JSON dict
        # like {"text": "...", "error_code": 0, ...}. Usiamo l'ultimo 'text'.
        for chunk in resp.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
            if not chunk:
                continue
            try:
                data = json.loads(chunk.decode("utf-8"))
            except Exception:
                continue

            # Se il worker segnala errore, abortiamo
            if data.get("error_code", 0) != 0:
                last_text = None
                break

            text = data.get("text")
            if isinstance(text, str):
                last_text = text

        if not last_text:
            return None

        full = last_text.strip()

        # Alcuni modelli possono aggiungere ```json ... ```: ripulisci
        for token in ("```json", "```JSON", "```"):
            if token in full:
                full = full.replace(token, "")
        full = full.strip()

        # Prova a fare parse diretto della stringa come JSON
        try:
            raw = json.loads(full)
        except Exception:
            # fallback: prendi l'ultimo blocco {...}
            start = full.rfind("{")
            end = full.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                raw = json.loads(full[start : end + 1])
            except Exception:
                return None

        if not isinstance(raw, dict):
            return None

        # Normalizza tipi e range così build_concept_bank può fidarsi
        present_val = raw.get("present", False)
        if isinstance(present_val, str):
            present = present_val.strip().lower() in ("true", "yes", "y", "present", "1")
        else:
            present = bool(present_val)

        try:
            confidence = float(raw.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "concept": raw.get("concept", concept_name),
            "present": present,
            "confidence": confidence,
            "rationale": raw.get("rationale", ""),
        }
