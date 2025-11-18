# generator.py
import os
import json
import requests
from typing import Any

# ---------------------------------------------------------
# LOAD GEMINI KEY (Streamlit Cloud + Local Support)
# ---------------------------------------------------------
def get_gemini_credentials():
    """
    Load Gemini API credentials in this priority:
    1. Streamlit Secrets (Streamlit Cloud + local)
    2. Environment variables (Local dev)
    """

    api_key = None
    model = None

    # 1) Try Streamlit secrets (works only inside Streamlit runtime)
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]

        if "GEMINI_MODEL" in st.secrets:
            model = st.secrets["GEMINI_MODEL"]

    except Exception:
        pass  # Streamlit not available (CLI environment)

    # 2) Fallback: environment variables
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not model:
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    return api_key, model


# ---------------------------------------------------------
# GOOGLE GENERATIVE API (v1beta)
# ---------------------------------------------------------
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def build_endpoint(model: str):
    return f"{BASE_URL}/{model}:generateContent"


def call_gemini_api(prompt: str, max_output_tokens: int = 500, temperature: float = 0.7) -> str:
    """
    Calls Google Gemini API using proper REST format.
    Supports Streamlit Cloud + local execution.
    """
    api_key, model = get_gemini_credentials()

    if not api_key:
        raise RuntimeError(
            "❌ GEMINI_API_KEY missing.\n\n"
            "Add it in Streamlit Cloud:\n"
            "  Settings → Secrets → GEMINI_API_KEY=\"your-key\"\n\n"
            "Or add a local .streamlit/secrets.toml:\n"
            "  GEMINI_API_KEY = \"your-key\"\n"
        )

    endpoint = build_endpoint(model)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.95,
            "maxOutputTokens": max_output_tokens
        }
    }

    response = requests.post(
        endpoint,
        params={"key": api_key},
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(f"Gemini API Error {response.status_code}: {response.text}")

    data = response.json()

    # ---------------------------------------------------------
    # Robust text extraction (supports all Gemini formats)
    # ---------------------------------------------------------
    try:
        # Newer format: candidates → content → parts → text
        if "candidates" in data and data["candidates"]:
            cand = data["candidates"][0]

            # Standard content → parts structure
            if "content" in cand:
                parts = []
                for c in cand["content"]:
                    for p in c.get("parts", []):
                        if p.get("text"):
                            parts.append(p["text"])
                if parts:
                    return "\n".join(parts)

            # Older format: output → content → parts → text
            if "output" in cand:
                parts = []
                for out in cand["output"]:
                    for c in out.get("content", []):
                        for p in c.get("parts", []):
                            if "text" in p:
                                parts.append(p["text"])
                if parts:
                    return "\n".join(parts)

        # Fallback recursive search for any "text" fields
        def find_text(obj: Any):
            results = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() == "text" and isinstance(v, str):
                        results.append(v)
                    else:
                        results.extend(find_text(v))
            elif isinstance(obj, list):
                for elem in obj:
                    results.extend(find_text(elem))
            return results

        texts = find_text(data)
        if texts:
            return "\n".join(texts)

    except Exception:
        pass

    return json.dumps(data)
