# generator.py
import os
import json
import requests
from typing import Any

# -------------------------------------------------------------------------
# SAFE CONFIG LOADER (Works for Streamlit + CLI + Background Worker)
# -------------------------------------------------------------------------
def load_gemini_config():
    """
    Load API key + model from:
    1. st.secrets (when running in Streamlit)
    2. Environment variables
    """

    api_key = None
    model = None

    # 1) Try Streamlit secrets (only works during `streamlit run`)
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]

        if "GEMINI_MODEL" in st.secrets:
            model = st.secrets["GEMINI_MODEL"]

    except Exception:
        # Streamlit not running / secrets not available
        pass

    # 2) Fallback to environment variables
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not model:
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    return api_key, model


# -------------------------------------------------------------------------
# Build correct Google endpoint
# -------------------------------------------------------------------------
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def build_endpoint(model: str = None):
    return f"{BASE_URL}/{model}:generateContent"


# -------------------------------------------------------------------------
# MAIN API CALLER (Google Gemini)
# -------------------------------------------------------------------------
def call_gemini_api(prompt: str, max_output_tokens: int = 500, temperature: float = 0.7) -> str:
    """
    Calls the Gemini API using the correct REST payload format.
    Reads secrets from Streamlit or env variables.
    """

    api_key, model = load_gemini_config()

    if not api_key:
        raise RuntimeError(
            "❌ GEMINI_API_KEY not found.\n\n"
            "Add it to `.streamlit/secrets.toml`:\n"
            "GEMINI_API_KEY = \"your-key\"\n"
            "GEMINI_MODEL = \"gemini-2.0-flash\"\n\n"
            "Or export it in the same terminal:\n"
            "export GEMINI_API_KEY=\"your-key\"\n"
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

    # -----------------------------------------------------------------
    # TEXT EXTRACTION LOGIC (All Gemini variants supported)
    # -----------------------------------------------------------------
    try:
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]

            # Newer schema: candidate.content[list] -> parts[list]
            if "content" in candidate:
                texts = []
                for block in candidate["content"]:
                    for part in block.get("parts", []):
                        if "text" in part:
                            texts.append(part["text"])
                if texts:
                    return "\n".join(texts)

            # Older schema: candidate.output -> content -> parts -> text
            if "output" in candidate:
                texts = []
                for out in candidate["output"]:
                    for item in out.get("content", []):
                        for part in item.get("parts", []):
                            if "text" in part:
                                texts.append(part["text"])
                if texts:
                    return "\n".join(texts)

        # fallback: recursively search for any "text" fields
        def find_texts(obj: Any):
            found = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() == "text" and isinstance(v, str):
                        found.append(v)
                    else:
                        found.extend(find_texts(v))
            elif isinstance(obj, list):
                for element in obj:
                    found.extend(find_texts(element))
            return found

        texts = find_texts(data)
        if texts:
            return "\n".join(texts)

    except Exception:
        pass

    return json.dumps(data)
