import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from .data_loader import (
    load_label_map,
    load_recipient_embeddings,
    load_sender_embeddings,
)


def _import_ctx_embedder():
    """
    Import CtxEmbedder lazily to avoid hard dependency on fasttext during startup/tests.
    Falls back to a dummy embedder with zero vectors if fasttext or artifacts are missing.
    """
    try:
        from src.preprocess._x_ctx_online import CtxEmbedder  # type: ignore
        return CtxEmbedder
    except Exception:
        class DummyCtx:
            def __init__(self, *args, **kwargs):
                self.dim = 128

            def embed_batch(self, df):
                return np.zeros((len(df), self.dim), dtype=np.float32)

        return DummyCtx


def _import_tri_tower():
    from src.train_clf._main_clf import TriTowerClassifier  # type: ignore
    return TriTowerClassifier


@lru_cache()
def get_ctx_embedder():
    Ctx = _import_ctx_embedder()
    try:
        return Ctx(model_dir=ROOT / "models")  # type: ignore[arg-type]
    except TypeError:
        return Ctx()


@lru_cache()
def get_tri_tower_model(d_in: int, n_classes: int):
    TriTowerClassifier = _import_tri_tower()
    model = TriTowerClassifier(d_in=d_in, n_classes=n_classes)
    state = torch.load(ROOT / "models" / "tri_tower_classifier.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _build_ctx_embedding(
    amount: float, tranx_type: str, msg: str, sender_id: str, recipient_id: str
) -> np.ndarray:
    embedder = get_ctx_embedder()
    now = pd.Timestamp.utcnow()
    df = pd.DataFrame(
        [
            {
                "txn_id": "demo_txn",
                "txn_time_utc": now,
                "amount": amount,
                "tranx_type": tranx_type,
                "channel": "MOBILE",
                "msg_content": msg,
                "sender_id": sender_id,
                "recipient_entity_id": recipient_id,
            }
        ]
    )
    emb = embedder.embed_batch(df)
    return emb  # shape (1, d_ctx)


def _pick_sender_embedding(sender_id: str) -> np.ndarray:
    snd_df, snd_cols = load_sender_embeddings()
    row = snd_df[snd_df["sender_id"] == sender_id].head(1)
    if row.empty:
        raise ValueError(f"Sender {sender_id} not found in offline features (date=ALL)")
    return row[snd_cols].to_numpy(dtype=np.float32)


def _pick_recipient_embedding(recipient_id: str) -> np.ndarray:
    rcv_df, rcv_cols = load_recipient_embeddings()
    row = rcv_df[rcv_df["recipient_entity_id"] == recipient_id].head(1)
    if row.empty:
        raise ValueError(f"Recipient {recipient_id} not found in offline features (date=ALL)")
    return row[rcv_cols].to_numpy(dtype=np.float32)


def model_infer(
    amount: float,
    msg: str,
    sender_id: str,
    recipient_id: str,
    tranx_type: str = "transfer_in",
) -> Dict:
    """
    Run tri-tower classifier using ctx + sender + recipient embeddings.
    Returns dict {success, label, confidence, raw_logit}
    """
    label_map = load_label_map()
    n_classes = len(label_map)

    ctx_emb = _build_ctx_embedding(amount, tranx_type, msg, sender_id, recipient_id)
    snd_emb = _pick_sender_embedding(sender_id)
    rcv_emb = _pick_recipient_embedding(recipient_id)

    x = np.concatenate([ctx_emb, snd_emb, rcv_emb], axis=1)
    model = get_tri_tower_model(d_in=x.shape[1], n_classes=n_classes)
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

    # Map index -> label code
    inv_index = {v["index"]: k for k, v in label_map.items()}
    label = inv_index.get(idx, "unknown")

    return {"success": True, "label": label, "confidence": confidence}
