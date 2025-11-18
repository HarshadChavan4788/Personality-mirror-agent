# generator.py
import os
import json
import requests
from typing import Any

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    # Do not raise at import so other tooling can import; raise when called if needed.
    pass

# Default model — change via env var if needed
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def build_endpoint(model: str = None) -> str:
    model = model or GEMINI_MODEL
    return f"{BASE_URL}/{model}:generateContent"


def call_gemini_api(prompt: str, max_output_tokens: int = 500, temperature: float = 0.7) -> str:
    """
    Call Google Generative Language REST API and return the generated text.
    Accepts max_output_tokens to be compatible with callers that pass that name.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Export it or place it in your .env")

    endpoint = build_endpoint()

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.95,
            "maxOutputTokens": max_output_tokens
        }
    }

    resp = requests.post(endpoint, params={"key": GEMINI_API_KEY}, json=payload, timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API Error {resp.status_code}: {resp.text}")

    data = resp.json()

    # Try multiple common shapes to extract text
    try:
        # primary: candidates -> content -> parts -> text
        if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
            cand = data["candidates"][0]
            # check cand.content -> parts
            if isinstance(cand.get("content"), list):
                parts_texts = []
                for item in cand["content"]:
                    if isinstance(item, dict) and item.get("parts"):
                        for p in item["parts"]:
                            if isinstance(p, dict) and p.get("text"):
                                parts_texts.append(p["text"])
                if parts_texts:
                    return "\n".join(parts_texts)
            # older shape: candidates[0].output -> content -> parts
            if isinstance(cand.get("output"), list):
                parts_texts = []
                for out in cand["output"]:
                    if isinstance(out, dict) and out.get("content"):
                        for c in out["content"]:
                            if isinstance(c, dict) and c.get("parts"):
                                for p in c["parts"]:
                                    if isinstance(p, dict) and p.get("text"):
                                        parts_texts.append(p["text"])
                if parts_texts:
                    return "\n".join(parts_texts)
        # fallback: search for any "text" keys in the response
        def find_texts(obj: Any):
            found = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() == "text" and isinstance(v, str):
                        found.append(v)
                    else:
                        found += find_texts(v)
            elif isinstance(obj, list):
                for e in obj:
                    found += find_texts(e)
            return found

        texts = find_texts(data)
        if texts:
            return "\n".join(texts)

    except Exception:
        pass

    # final fallback: return JSON dump
    return json.dumps(data)