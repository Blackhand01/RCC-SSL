#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client locale per LLaVA-Med v1.5 (o modelli compatibili) via Hugging Face.

- Usa l'interfaccia "image-text-to-text" standard:
  * messages (role/content) + apply_chat_template
  * AutoProcessor + LlavaForConditionalGeneration

- Contratto esterno compatibile col vecchio VLMClientHF:
    VLMClientHF(model_name, device=None, dtype="bfloat16", debug=None)
    .ask_concept(image_path, concept_name, base_prompt, temperature, max_new_tokens)

- NOTA: per coerenza con LLaVA-Med, la lingua dell'istruzione e inglese.
  Il default usa il checkpoint HF compatibile: chaoyinshe/llava-med-v1.5-mistral-7b-hf
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


ImageLike = Union[str, Path, Image.Image]


@dataclass
class ConceptResponse:
    concept: str
    question_text: str
    messages: List[Dict[str, Any]]
    raw_response: str
    parsed_label: str  # "Present" / "Absent" / "Uncertain" / "Unknown"


class VLMClientHF:
    """
    Client per modelli LLaVA-like in formato HF "image-text-to-text".

    Default: chaoyinshe/llava-med-v1.5-mistral-7b-hf (HF compatibile)
    """

    def __init__(
        self,
        model_name: str = "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        debug: Optional[bool] = None,
        mode: str = "describe",
    ) -> None:
        self.model_name = model_name
        # Respect the default 'describe' when mode is None/empty.
        mode = (mode or "describe").lower()
        if mode not in {"concept", "describe"}:
            raise ValueError(
                f"Unsupported VLM mode: {mode!r} (expected 'concept' or 'describe')"
            )
        self.mode = mode

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Dtype: BF16 consigliato per llava-med su GPU A40; FP32 su CPU
        if self.device == "cuda":
            if dtype.lower() == "float16":
                model_dtype = torch.float16
            elif dtype.lower() == "bfloat16":
                model_dtype = torch.bfloat16
            else:
                model_dtype = torch.float32
        else:
            model_dtype = torch.float32

        # Debug: env VLM_DEBUG=1 abilita logging verboso
        if debug is None:
            self.debug = os.getenv("VLM_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

        if self.debug:
            print(
                f"[VLM-HF] Loading model '{self.model_name}' "
                f"on device={self.device}, dtype={model_dtype} "
                "(LlavaForConditionalGeneration)"
            )

        # Processor + modello con supporto multimodale (LlavaForConditionalGeneration)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load AutoProcessor for {self.model_name}. "
                "Upgrade 'transformers' to a version that supports llava_mistral."
            ) from exc

        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=model_dtype,
                device_map=None,  # singola GPU/CPU; spostiamo noi su device
            ).to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LlavaForConditionalGeneration for {self.model_name}. "
                "Upgrade 'transformers' and ensure support for model type llava_mistral."
            ) from exc

        self.model.eval()

        # Fallback patch_size se mancante (vecchie versioni transformers)
        if getattr(self.processor, "patch_size", None) is None:
            vision_cfg = getattr(self.model.config, "vision_config", None)
            patch_size = getattr(vision_cfg, "patch_size", None)
            patch_size = patch_size or getattr(
                getattr(self.processor, "image_processor", None), "patch_size", None
            )
            if patch_size is None:
                patch_size = 14
                if self.debug:
                    print(
                        "[VLM-HF] 'patch_size' not found; using fallback 14 "
                        "(ViT-L/14-compatible)."
                    )
            self.processor.patch_size = patch_size

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_question_text(self, concept_name: str, base_prompt: str) -> str:
        """
        Testo dell'istruzione lato utente.

        - In mode 'concept': classificatore Present/Absent/Uncertain (comportamento attuale).
        - In mode 'describe': descrizione libera, senza label categoriche.
        """
        if self.mode == "describe":
            return (
                "Carefully examine the attached patch."
                "Which histologic concepts you can clearly identify?"
                "Write a concise but information-dense answer."
            )
                
        return (
            "You are a board-certified renal pathologist. Act as a classifier and determine"
            " whether the specified histologic concept is present in the attached patch.\n\n"
            "Task: Examine only the attached histology patch:\n"
            "{base_prompt}\n\n"
            "Instructions:\n"
            " - First line: Decide if the concept, in the image, is or Uncertain or Present or Absent.\n"
            " - Second line: provide a concise, information-dense explanation (one short sentence or phrase).\n"
        )

    def _build_messages(
        self, concept_name: str, base_prompt: str
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Costruisce la conversazione nello standard HF:
        messages = [{role: 'user', content: [{type: 'image'}, {type: 'text', ...}]}]
        """
        question_text = self._build_question_text(concept_name, base_prompt)
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        return messages, question_text

    # ------------------------------------------------------------------
    # Parsing della label
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_label(raw: str) -> str:
        """
        Mappa la risposta in {Present, Absent, Uncertain, Unknown}
        in modo ultra-conservativo:

        - accetta solo risposte "pulite":
            Present / Absent / Uncertain / Unknown (con o senza punteggiatura finale)
            Yes / No -> mappate su Present / Absent
        - qualunque output discorsivo viene marcato Unknown.
        """
        if not raw:
            return "Unknown"

        text = raw.strip()
        if not text:
            return "Unknown"

        # usa solo la prima riga, per evitare prompt/templating residui
        first_line = text.splitlines()[0].strip()
        first_line = first_line.strip(" \"'")
        lower = first_line.lower()

        m = re.match(r"^(present|absent|uncertain|unknown)[\\.!?]*$", lower)
        if m:
            token = m.group(1)
            if token == "present":
                return "Present"
            if token == "absent":
                return "Absent"
            if token == "uncertain":
                return "Uncertain"
            return "Unknown"

        m_yn = re.match(r"^(yes|no)[\\.!?]*$", lower)
        if m_yn:
            return "Present" if m_yn.group(1) == "yes" else "Absent"

        return "Unknown"

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------
    def ask_concept(
        self,
        image_path: ImageLike,
        concept_name: str,
        base_prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 32,
    ) -> Optional[Dict[str, Any]]:
        """
        Esegue una singola query concept-level.

        NOTE:
        - In mode 'concept' behaves as classifier.
        - In mode 'describe' parsed_label will typically be 'Unknown'; the free-form
          description lives in raw_response.

        Ritorna un dict serializzabile (usato da build_concept_bank.py):
        {
          "concept": str,
          "question_text": str,
          "messages": [...],
          "raw_response": str,
          "parsed_label": str
        }
        """

        # Carica immagine
        if isinstance(image_path, (str, Path)):
            image_path = Path(image_path)
            if not image_path.is_file():
                if self.debug:
                    print(f"[VLM-HF] Image not found: {image_path}")
                return None
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")
        else:
            raise TypeError(f"Unsupported image_path type: {type(image_path)}")

        messages, question_text = self._build_messages(concept_name, base_prompt)

        if self.debug:
            print(
                f"[VLM-HF] >>> REQUEST\n"
                f"concept={concept_name}\n"
                f"image={image_path}\n"
                f"question_text (full)={question_text!r}\n"
                f"{'-'*60}"
            )

        # Applichiamo il chat template del modello (standard LLaVA / HF)
        prompt: str
        processor_has_template = getattr(self.processor, "chat_template", None) not in (
            None,
            "",
        )

        if hasattr(self.processor, "apply_chat_template") and processor_has_template:
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
        else:
            tok = getattr(self.processor, "tokenizer", None)
            tok_has_template = (
                tok is not None
                and getattr(tok, "chat_template", None) not in (None, "")
                and hasattr(tok, "apply_chat_template")
            )

            if tok_has_template:
                prompt = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback minimale stile USER/ASSISTANT con immagine
                if self.debug:
                    print(
                        "[VLM-HF] No chat_template on processor/tokenizer; "
                        "falling back to manual single-turn prompt."
                    )

                user_text_parts = []
                for m in messages:
                    if m.get("role") != "user":
                        continue
                    for item in m.get("content", []):
                        if item.get("type") == "text":
                            user_text_parts.append(item.get("text", ""))

                user_text = "\n".join(user_text_parts)
                prompt = f"USER: <image>\n{user_text}\nASSISTANT:"

        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
        }
        if temperature and temperature > 0.0:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=0.9,
                )
            )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        gen_ids = generated_ids[:, prompt_len:]

        decoded = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True
        )
        raw = decoded[0].strip() if decoded else ""

        label = self._parse_label(raw)

        if self.debug:
            print(
                f"[VLM-HF] <<< RESPONSE concept={concept_name}\n"
                f"raw={raw!r}\nparsed_label={label}\n{'='*80}"
            )

        # Dict serializzabile (messages -> JSON piu avanti)
        return {
            "concept": concept_name,
            "question_text": question_text,
            "messages": messages,
            "raw_response": raw,
            "parsed_label": label,
        }


if __name__ == "__main__":
    # Smoke test minimale (solo per debug ambienti, non per produzione)
    print(json.dumps({"status": "vlm_client_hf_ok"}))
