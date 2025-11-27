import json
import os
from typing import Dict, Optional

import google.generativeai as genai

from .data_loader import load_label_map

# Allow override via env; default to a stable public model (list_models confirmed)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-lite-latest")
# Confidence threshold for accepting LLM result
CONF_THRESHOLD = float(os.getenv("GEMINI_CONF_THRESHOLD", 0.8))


def _safe_parse(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except Exception:
        # try to extract JSON block
        if "{" in text and "}" in text:
            snippet = text[text.find("{") : text.rfind("}") + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
    return None


def call_gemini(message: str, recipient: str) -> Dict:
    """
    Call Gemini to classify. Returns dict {success, label, confidence, raw}
    If SKIP_GEMINI env is set, returns unknown quickly for tests.
    """
    if os.getenv("SKIP_GEMINI"):
        return {"success": False, "label": "unknown", "confidence": 0.0, "raw": "SKIPPED"}

    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    label_map = load_label_map()
    labels = [k for k in label_map.keys() if k != "OTH"]

    candidates = []
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        candidates.append(env_model)
    if DEFAULT_MODEL not in candidates:
        candidates.append(DEFAULT_MODEL)
    # Fallbacks based on list_models (generateContent supported)
    fallback_models = [
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.0-flash",
        "models/gemini-pro-latest",
    ]
    for m in fallback_models:
        if m not in candidates:
            candidates.append(m)

    prompt = f"""
You are a transaction classifier. Choose exactly one of these labels (exclude OTH): {labels}.
Input fields:
- message: \"{message}\"
- recipient: \"{recipient}\"

Respond ONLY in JSON: {{"label": "<one_of_labels_or_unknown>", "confidence": <0-1 float>}}.
If you are not confident, set label to "unknown".
If confidence < {CONF_THRESHOLD} -> label must be "unknown".
"""
    last_err = None
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt, generation_config={"temperature": 0.2})
            text = resp.text or ""
            parsed = _safe_parse(text)
            if not parsed or "label" not in parsed:
                last_err = f"parse_failed_{model_name}"
                continue
            label = parsed.get("label", "unknown")
            conf = float(parsed.get("confidence", 0.0) or 0.0)
            if conf < CONF_THRESHOLD:
                label = "unknown"
            return {"success": label != "unknown", "label": label, "confidence": conf, "raw": text, "model": model_name}
        except Exception as exc:  # pragma: no cover - network/availability issues
            last_err = f"{model_name}: {exc}"
            continue

    return {"success": False, "label": "unknown", "confidence": 0.0, "raw": f"All models failed: {last_err}"}
