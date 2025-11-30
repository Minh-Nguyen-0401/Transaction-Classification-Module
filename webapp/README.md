# Tranx Classifier Web Demo

## Overview

This `webapp/` folder contains a small FastAPI-based demo that exposes the trained transaction classification pipeline as:

1. **A REST API** for programmatic access.
2. **A minimal HTML frontend** for interactive experimentation.

Each classification request goes through **three decision layers**:

1. **Rule layer** – heuristic mapping from cleaned text / transaction type to labels.
2. **LLM layer (Gemini)** – calls Google Gemini to classify, with confidence-based fallback.
3. **Model layer (Tri‑tower classifier)** – uses the offline embeddings + online context encoder from the main pipeline.

If a layer is confident enough, it decides early; otherwise the request falls through to the next layer.

The webapp reuses artifacts produced by the training pipeline in the root project:

- Context encoder & helpers (from `src/preprocess/_x_ctx_online.py` and `models/`).
- Sender and recipient embeddings (from `data/feature_store/`).
- Trained tri-tower classifier weights (from `models/tri_tower_classifier.pt`).
- Label map (from `data/label_map.json`).

---

## Project Layout (webapp/)

- `webapp/main.py`
  - FastAPI application factory and routing.
  - Endpoints: `/health`, `/options`, `/classify`, and `/` (frontend HTML).
- `webapp/schemas.py`
  - Pydantic models for request/response payloads.
- `webapp/services/rule_layer.py`
  - String normalization (accent stripping, lowercasing).
  - Simple keyword and transaction-type based classification rules.
- `webapp/services/llm_layer.py`
  - Thin client for Google Gemini.
  - Builds prompt, calls the model, parses JSON output, enforces a confidence threshold.
- `webapp/services/model_layer.py`
  - Loads label map and offline sender/recipient embeddings.
  - Loads context encoder and tri-tower classifier.
  - Builds a single-row dataframe for the current transaction and runs inference.
- `webapp/services/data_loader.py`
  - Lazily loads `data/label_map.json`.
  - Lazily loads sender and recipient embeddings from `data/feature_store/...`.
  - Provides default sender/recipient IDs and dropdown options for the frontend.
- `webapp/frontend/index.html`
  - Single-page HTML+JS frontend.
  - Calls `/options` to populate dropdowns and `/classify` to run the pipeline.
- `webapp/tests/`
  - API tests using `TestClient`, with the ability to skip real Gemini calls.
- `webapp/EXAMPLES.md`
  - Example payloads and scenarios for manual testing.

---

## Dependencies

The webapp has its own `requirements.txt`:

```bash
pip install -r webapp/requirements.txt
```

Key dependencies:

- **FastAPI**, **uvicorn** – web framework + ASGI server.
- **google-generativeai** – Gemini client.
- **pandas**, **torch**, **scikit-learn**, **joblib**, **fasttext-wheel** – shared with the main pipeline for feature and model loading.
- **python-multipart**, **jinja2**, **httpx** – form handling / templating / HTTP client utilities.

You also need the **base project dependencies** listed in the root `requirements.txt` and the trained artifacts described below.

---

## Required Artifacts & Data

To run the full 3-layer flow, you need the following files produced by the training pipeline:

- **Label map**
  - `data/label_map.json` – maps label codes (e.g. `FOO`, `BIL`, `SHP`, `OTH`) to indices and metadata.

- **Context encoder**
  - Model weights and preprocessing artifacts under `models/`.
  - Used indirectly via `src/preprocess/_x_ctx_online.py`.

- **Sender embeddings (offline)**
  - `data/feature_store/x_snd/date=ALL/x_snd_user_state.parquet`
  - Must contain `sender_id` and `snd_emb_*` columns.

- **Recipient embeddings (offline)**
  - `data/feature_store/x_rcv/date=ALL/x_rcv_entity_state.parquet`
  - Must contain `recipient_canon_id` / `recipient_entity_id` and `rcv_emb_*` columns.

- **Tri-tower classifier weights**
  - `models/tri_tower_classifier.pt` – trained multi-layer classifier combining context, sender and recipient embeddings.

If these artifacts are missing, the model layer may fail or fall back to dummy embeddings.

---

## Configuration & Environment Variables

The webapp behavior is controlled via several environment variables:

- `GEMINI_API_KEY` (required for real LLM calls)
  - Google Gemini API key. Needed for the LLM layer to function.

- `GEMINI_MODEL` (optional)
  - Overrides the default model name (e.g. `models/gemini-flash-lite-latest`).
  - If not set, the code picks a reasonable default and then tries a list of fallbacks.

- `GEMINI_CONF_THRESHOLD` (optional, default `0.8`)
  - Minimum confidence for accepting an LLM result.
  - If the parsed confidence is below this threshold, the LLM label is forced to `"unknown"` and the request falls through to the model layer.

- `SKIP_GEMINI` (optional)
  - If set to any value, the LLM layer is **disabled**.
  - `call_gemini` will immediately return `{label: "unknown", success: false}` so that the request falls back to the model layer.
  - Useful for offline development, unit tests, or environments without outbound network access.

Make sure the working directory when running the app is the repository root, so that relative paths like `data/...` and `models/...` resolve correctly.

---

## Running the Web App

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r webapp/requirements.txt
   ```

2. **Ensure artifacts exist**

   - Run the training pipeline (see root `README.md`) to produce encoder models, feature store parquet files and the classifier checkpoint.
   - Verify that `data/label_map.json`, `data/feature_store/x_snd/date=ALL/x_snd_user_state.parquet`, `data/feature_store/x_rcv/date=ALL/x_rcv_entity_state.parquet`, and `models/tri_tower_classifier.pt` exist.

3. **Set environment variables**

   - For full 3-layer behavior (including LLM):

     ```bash
     set GEMINI_API_KEY=your_api_key_here   # Windows cmd
     # or
     $env:GEMINI_API_KEY="your_api_key_here"   # PowerShell
     ```

   - For offline / no-network mode:

     ```bash
     set SKIP_GEMINI=1
     ```

4. **Start the server**

   From the repository root:

   ```bash
   uvicorn webapp.main:app --reload
   ```

5. **Open the UI**

   - Visit <http://127.0.0.1:8000/> in your browser.
   - The page will:
     - Call `GET /options` to fetch sender/recipient dropdowns.
     - Submit your form to `POST /classify` and display per-layer status and final decision.

---

## API Reference

### `GET /health`

- Simple health check.
- **Response:** `{ "status": "ok" }` if the app is up.

### `GET /options`

- Returns allowed sender and recipient IDs for the frontend.
- **Query params:**
  - `limit` (int, optional, default 50) – maximum number of records per list.
- **Response model:**
  - `default_sender` (string or null)
  - `default_recipient` (string or null)
  - `senders`: list of `{ code, display_name }`
  - `recipients`: list of `{ code, display_name }`

### `POST /classify`

- Main classification endpoint.
- **Request body (`ClassifyRequest`):**
  - `message` (string, required) – transaction message/content.
  - `amount` (float, required, > 0) – amount to transfer.
  - `recipient_entity_id` (string, required).
  - `sender_id` (string, optional) – if omitted, defaults to `get_default_sender()`.
  - `tranx_type` (string, optional) – optional hint for the rule layer.

- **Response body (`ClassifyResponse`):**
  - `layer1`, `layer2`, `layer3` – each a `LayerResult` with:
    - `status`: `"success" | "fail" | "skipped"`.
    - `label`: label code or `null`.
    - `confidence`: float or `null`.
    - `detail`: extra text (e.g. rule match, LLM raw output, error messages).
  - `final_label`: label code chosen by the pipeline.
  - `final_confidence`: numeric confidence (usually from the deciding layer).
  - `decided_by`: one of `"layer1_rule"`, `"layer2_llm"`, `"layer3_model"`, or `"fallback_oth"`.

### `GET /`

- Serves the static HTML UI from `webapp/frontend/index.html`.
- No parameters; returns the classification demo page.

---

## Inference Flow (Detailed)

1. **Normalize input**
   - Frontend collects `message`, `amount`, `recipient_entity_id`, and an optional `sender_id`.
   - Backend fills in defaults for sender/recipient if missing.

2. **Layer 1 – Rule-based classification**
   - `rule_classify` cleans the message (lowercase, remove accents) and attempts:
     - Mappings from `tranx_type` → label (via `TRANX_TYPE_MAP`).
     - Keyword-based matches for each label code (via `KEYWORDS`).
   - On match, the pipeline stops here and returns the rule-based label.

3. **Layer 2 – LLM (Gemini)**
   - If rules fail, `call_gemini` is invoked (unless `SKIP_GEMINI` is set).
   - The function:
     - Builds a prompt listing all non-`OTH` labels.
     - Asks Gemini to output JSON `{label, confidence}`.
     - Parses the JSON, enforces `GEMINI_CONF_THRESHOLD` and treats low confidence as `"unknown"`.
   - If a confident non-`unknown` label is produced, the pipeline stops here.

4. **Layer 3 – Tri-tower model**
   - If LLM fails or is skipped, `model_infer` runs:
     - Builds a one-row dataframe with basic transaction fields (amount, type, channel, message, sender, recipient, timestamp=now).
     - Uses `CtxEmbedder` (from `src/preprocess/_x_ctx_online.py`) to create a context embedding.
     - Looks up sender and recipient embeddings from `data/feature_store/...` using the provided IDs.
     - Concatenates context + sender + recipient vectors and feeds them into `TriTowerClassifier` loaded from `models/tri_tower_classifier.pt`.
     - Applies softmax to obtain probabilities, picks the argmax index and maps it back to a label using `data/label_map.json`.

5. **Output aggregation**
   - The app returns all three layer results plus the final decision and which layer decided it.

---

## Testing Notes

- To run tests (for example via `pytest`):
  - Set `SKIP_GEMINI=1` to avoid external network calls.
  - Ensure that required data and models exist, or skip tests that require them.
- The `webapp/tests` folder focuses on:
  - Basic health and options endpoints.
  - `POST /classify` happy paths with rule or model decisions.

---

## Relationship to the Main Pipeline

- The main pipeline (see root `README.md` and `data/DATA_FLOW.md`) is responsible for **training** encoders and the tri-tower classifier.
- This `webapp/` package provides a convenient **inference and demo layer** on top of those artifacts.
- If you retrain or change the underlying models:
  - Regenerate artifacts and make sure they are copied/overwritten under `models/` and `data/feature_store/`.
  - The webapp will automatically pick up the new versions on restart.
