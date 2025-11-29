#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to test VLM client HTTP calls directly.
Run this locally with one image to verify the worker is responding correctly.
"""

import base64
import json
import os
from pathlib import Path

import requests

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://localhost:10000")
MODEL_NAME = "llava-med-v1.5-mistral-7b"

def to_b64(image_path: Path) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def test_llava_call(image_path: Path, concept_name: str, base_prompt: str):
    """Test a single VLM call and print raw response."""
    image_b64 = to_b64(image_path)

    prompt = f"""<image>
You are a board-certified renal pathologist.

Analyse the attached histology patch.

Concept: {concept_name}
Question: {base_prompt}

Respond ONLY with a single line containing a valid JSON object with the following keys:
- "concept": string
- "present": boolean (true or false)
- "confidence": float between 0 and 1
- "rationale": string (max 20 words)

Example:
{{"concept": "Clear cytoplasm (ccRCC)", "present": true, "confidence": 0.73, "rationale": "Concise reason"}}

Now return ONLY the JSON object, with no additional text before or after."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "temperature": 0.2,
        "max_new_tokens": 128,
        # mai None: evita crash del worker su KeywordsStoppingCriteria
        "stop": "###",
    }

    print(f"Testing with image: {image_path}")
    print(f"Concept: {concept_name}")
    print(f"Sending request to: {CONTROLLER_URL}/worker_generate_stream")

    try:
        r = requests.post(f"{CONTROLLER_URL}/worker_generate_stream", json=payload, stream=True, timeout=120)
        print(f"STATUS: {r.status_code}")

        if r.status_code != 200:
            print(f"ERROR RESPONSE: {r.text}")
            return

        text = ""
        for chunk in r.iter_lines(delimiter=b"\0"):
            if chunk:
                try:
                    data = json.loads(chunk.decode())
                    text = data.get("text", "")
                except json.JSONDecodeError:
                    continue

        print(f"TEXT FIELD (first 400 chars): {repr(text[:400])}")

        # Try to parse as JSON
        try:
            parsed = json.loads(text.strip())
            print("PARSED JSON:", json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"JSON PARSE ERROR: {e}")

    except Exception as e:
        print(f"HTTP ERROR: {e}")

if __name__ == "__main__":
    # Use first available image from concept candidates
    images_root = Path("concept_candidates_images")
    if not images_root.exists():
        print(f"Images directory {images_root} not found. Run build_concept_candidates.py first.")
        exit(1)

    # Find first image
    image_files = list(images_root.rglob("*.png"))
    if not image_files:
        print("No PNG images found in concept_candidates_images/")
        exit(1)

    test_image = image_files[0]
    concept_name = "Clear cytoplasm (ccRCC)"
    base_prompt = "Identify viable renal tumour cells with abundant optically clear or glassy cytoplasm and sharp cell borders, in keeping with clear cell renal cell carcinoma. Exclude adipocytes, stromal fat, artefactual perinuclear clearing, and foamy macrophages."

    test_llava_call(test_image, concept_name, base_prompt)
