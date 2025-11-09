# Data Flow in Transaction Classification Pipeline

## Overview

This document explains how data flows through the entire pipeline from synthetic generation to classifier training.

## Data Generation

**Command:**
```powershell
python data/generate_synthetic_data.py --num-transactions 100 --num-days 5
```

**Generates:**
- `transactions_landing.parquet` - Main transaction table **with category labels already included**
- `customer_profile.parquet` - Sender profiles
- `recipient_entity.parquet` - Recipient entities
- `merchant_profile.parquet` - Merchant details
- `person_profile.parquet` - Person profiles
- `recipient_alias.parquet` - Recipient aliases
- `train_supervised.csv` - Training labels (txn_id, sender_id, recipient_entity_id, txn_date, label_l1)

## Pipeline Flow

### Step 1: Train Encoders

**Context Encoder:**
```powershell
python src/_build_enc/train_ctx_enc.py
```
- **Input:** `data/raw/transactions_landing.parquet`
- **Output:** `models/ctx_encoder.pt`, `models/ctx_*.pkl` (scalers, one-hot encoders)
- **Purpose:** Learn to encode transaction context (amount, time, type, channel, message)

**Sender Encoder:**
```powershell
python src/_build_enc/train_snd_enc.py
```
- **Input:** `data/raw/transactions_landing.parquet`
- **Output:** `models/snd_encoder.pt`, `models/ctx_*.pkl`
- **Purpose:** Learn to encode sender transaction sequences

### Step 2: Generate Features

**Context Features (Online):**
```powershell
python src/preprocess/_x_ctx_online.py
```
- **Input:** `data/raw/transactions_landing.parquet`
- **Output:** `data/processed/ctx_emb.parquet`
- **Contains:** `txn_id` + `ctx_emb_1` to `ctx_emb_128`
- **Note:** Uses the trained context encoder to embed each transaction

**Sender Features (Offline):**
```powershell
python src/preprocess/_x_snd_offline.py
```
- **Input:** `data/raw/transactions_landing.parquet`
- **Output:** 
  - Daily snapshots: `data/feature_store/x_snd/date=YYYY-MM-DD/x_snd_user_state.parquet`
  - Cumulative: `data/feature_store/x_snd/date=ALL/x_snd_user_state.parquet`
- **Contains:** `sender_id` + `snd_emb_1` to `snd_emb_128` + metadata
- **Auto-processes:** All unique transaction dates (skips existing)

**Recipient Features (Offline):**
```powershell
python src/preprocess/_x_rcv_offline.py
```
- **Input:** 
  - `data/raw/transactions_landing.parquet`
  - `data/raw/recipient_entity.parquet`
  - `data/raw/merchant_profile.parquet`
  - `data/raw/person_profile.parquet`
- **Output:**
  - Daily snapshots: `data/feature_store/x_rcv/date=YYYY-MM-DD/x_rcv_entity_state.parquet`
  - Cumulative: `data/feature_store/x_rcv/date=ALL/x_rcv_entity_state.parquet`
- **Contains:** `recipient_canon_id` + `rcv_emb_1` to `rcv_emb_96` + metadata
- **Auto-processes:** All unique transaction dates (skips existing)

### Step 3: Train Classifier

```powershell
python src/train_clf/train_main_clf.py
```

**Inputs:**
1. `data/raw/train_supervised.csv` - Labels (txn_id, sender_id, recipient_entity_id, txn_date, label_l1)
2. `data/processed/ctx_emb.parquet` - Context embeddings
3. `data/feature_store/x_snd/date=*/x_snd_user_state.parquet` - Sender embeddings (by date)
4. `data/feature_store/x_rcv/date=*/x_rcv_entity_state.parquet` - Recipient embeddings (by date)

**Merge Logic:**
```python
# Load labels
train_df = pd.read_csv("train_supervised.csv")
# txn_id, sender_id, recipient_entity_id, txn_date, label_l1

# Merge context embeddings (by txn_id)
train_df = train_df.merge(ctx_emb, on="txn_id")

# Calculate feature date (with lookback to avoid data leak)
train_df["_feature_date"] = (train_df["txn_date"] - lookback_days).strftime("%Y-%m-%d")

# Merge sender embeddings (by sender_id + _feature_date)
train_df = train_df.merge(snd_features, on=["sender_id", "_feature_date"])

# Merge recipient embeddings (by recipient_entity_id + _feature_date)
train_df = train_df.merge(rcv_features, on=["recipient_entity_id", "_feature_date"])

# Extract label indices from label_map.json
y = train_df["label_l1"].map(label_to_index)
```

**Output:**
- `models/tri_tower_classifier.pt` - Trained classifier
- `models/cls_label_map.pkl` - Label mapping for inference

## Key Points

### Why train_supervised.csv exists separately?

Even though `transactions_landing.parquet` contains the category labels:
- **train_supervised.csv** is a clean, focused dataset for training
- Contains only necessary columns: txn_id, sender_id, recipient_entity_id, txn_date, label_l1
- Easier to manage train/test splits
- Separates "raw data" from "training data"

### Lookback Days (Temporal Consistency)

The classifier training uses `lookback_days=1` by default:
- For a transaction on 2025-11-05, it uses features from 2025-11-04
- This prevents **data leakage** (using future information)
- In production: Use yesterday's features to classify today's transactions

### Feature Store Structure

```
data/feature_store/
├── x_snd/
│   ├── date=2025-11-01/
│   │   └── x_snd_user_state.parquet
│   ├── date=2025-11-02/
│   │   └── x_snd_user_state.parquet
│   └── date=ALL/
│       └── x_snd_user_state.parquet
└── x_rcv/
    ├── date=2025-11-01/
    │   └── x_rcv_entity_state.parquet
    ├── date=2025-11-02/
    │   └── x_rcv_entity_state.parquet
    └── date=ALL/
        └── x_rcv_entity_state.parquet
```

**Daily snapshots** allow point-in-time lookups for training.
**Cumulative (date=ALL)** stores latest state for all entities.

## Complete Example Run

```powershell
# 1. Generate data (100 transactions over 5 days)
cd data
python generate_synthetic_data.py --num-transactions 100 --num-days 5

# 2. Train encoders
cd ..
python src/_build_enc/train_ctx_enc.py
python src/_build_enc/train_snd_enc.py

# 3. Generate features
python src/preprocess/_x_ctx_online.py
python src/preprocess/_x_snd_offline.py
python src/preprocess/_x_rcv_offline.py

# 4. Train classifier
python src/train_clf/train_main_clf.py

# 5. Model ready at models/tri_tower_classifier.pt
```
