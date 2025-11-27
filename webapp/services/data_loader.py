import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@lru_cache()
def load_label_map() -> Dict[str, Dict]:
    path = ROOT / "data" / "label_map.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@lru_cache()
def load_sender_embeddings() -> Tuple[pd.DataFrame, List[str]]:
    path = ROOT / "data" / "feature_store" / "x_snd" / "date=ALL" / "x_snd_user_state.parquet"
    df = pd.read_parquet(path)
    df["sender_id"] = df["sender_id"].astype(str)
    emb_cols = [c for c in df.columns if c.startswith("snd_emb_")]
    # keep latest/first per sender_id to ensure one row per sender
    df_unique = df.drop_duplicates(subset=["sender_id"])
    return df_unique[["sender_id"] + emb_cols], emb_cols


@lru_cache()
def load_recipient_embeddings() -> Tuple[pd.DataFrame, List[str]]:
    path = ROOT / "data" / "feature_store" / "x_rcv" / "date=ALL" / "x_rcv_entity_state.parquet"
    df = pd.read_parquet(path)
    df["recipient_entity_id"] = df["recipient_canon_id"].astype(str)
    emb_cols = [c for c in df.columns if c.startswith("rcv_emb_")]
    df = df.drop_duplicates(subset=["recipient_entity_id"])
    df = df[["recipient_entity_id"] + emb_cols]
    return df, emb_cols


@lru_cache()
def load_recipient_names() -> Dict[str, str]:
    """Map recipient_entity_id -> primary_display_name if available."""
    path = ROOT / "data" / "raw" / "recipient_entity.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return dict(zip(df["recipient_entity_id"].astype(str), df["primary_display_name"]))
    return {}


def get_default_sender() -> Optional[str]:
    snd_df, _ = load_sender_embeddings()
    return snd_df["sender_id"].iloc[0] if not snd_df.empty else None


def get_default_recipient() -> Optional[str]:
    rcv_df, _ = load_recipient_embeddings()
    return rcv_df["recipient_entity_id"].iloc[0] if not rcv_df.empty else None


def list_senders(limit: int = 50) -> List[Dict[str, str]]:
    snd_df, _ = load_sender_embeddings()
    opts = snd_df[["sender_id"]].head(limit).to_dict(orient="records")
    return opts


def list_recipients(limit: int = 50) -> List[Dict[str, str]]:
    names = load_recipient_names()
    rcv_df, _ = load_recipient_embeddings()
    rows = []
    for _, row in rcv_df.head(limit).iterrows():
        rows.append(
            {
                "recipient_entity_id": row["recipient_entity_id"],
                "display_name": names.get(row["recipient_entity_id"], row["recipient_entity_id"]),
            }
        )
    return rows
