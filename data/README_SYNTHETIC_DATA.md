# Synthetic Data Generator

## Quick Start

### 1. Generate Small Test Dataset (for unit testing)
```powershell
cd data
python generate_synthetic_data.py --num-transactions 100 --num-days 5
```

This creates:
- 100 transactions over 5 days
- ~10 customers
- ~15 recipients (mix of merchants and persons)
- All supporting profile tables

### 2. Generate Full Dataset (for real training)
```powershell
python generate_synthetic_data.py --num-transactions 10000 --num-days 30
```

This creates:
- 10,000 transactions over 30 days
- ~500 customers
- ~667 recipients
- Realistic category distributions

## Generated Files

All files are saved to `data/raw/`:

| File | Description | Schema |
|------|-------------|--------|
| `transactions_landing.parquet` | Main transaction table | As per `transactions_landing` in data dictionary |
| `customer_profile.parquet` | Sender profiles | As per `customer_profile` |
| `recipient_entity.parquet` | Recipient registry | As per `recipient_entity` |
| `merchant_profile.parquet` | Merchant details | As per `merchant_profile` |
| `person_profile.parquet` | Person profiles | As per `person_profile` |
| `recipient_alias.parquet` | Recipient aliases | As per `recipient_alias` |
| `train_supervised.csv` | Training data with labels | For classifier training |

## Data Characteristics

### Realistic Distributions
- **Categories**: All 10 categories (BIL, FOO, TRN, HLT, INS, SHP, ENT, EDU, FIN, OTH)
- **Transaction Types**: 8 types mapped to categories
- **Channels**: MOBILE (60%), WEB (40%)
- **Amounts**: Log-normal distribution varying by category
- **Messages**: Vietnamese transaction messages

### Entity Relationships
- Merchants get MCCs matching their category
- Person-to-person transfers use PERSON entities
- Bill payments mostly use MERCHANT entities
- Proper foreign key relationships maintained

### Temporal Distribution
- Transactions spread across specified days
- Random times throughout each day
- Sorted chronologically

## Pipeline Execution Order

After generating data, run:

```powershell
# 1. Train context encoder
python src/_build_enc/train_ctx_enc.py

# 2. Train sender encoder  
python src/_build_enc/train_snd_enc.py

# 3. Generate context features
python src/preprocess/_x_ctx_online.py

# 4. Generate sender features (processes all dates automatically)
python src/preprocess/_x_snd_offline.py

# 5. Generate recipient features (processes all dates automatically)
python src/preprocess/_x_rcv_offline.py

# 6. Train classifier
python src/train_clf/train_main_clf.py
```

## Customization

### Change Date Range
```powershell
python generate_synthetic_data.py --start-date "2025-10-01" --num-days 60
```

### Adjust Scale
```powershell
# Tiny (debugging)
python generate_synthetic_data.py --num-transactions 50 --num-days 3

# Small (unit tests)
python generate_synthetic_data.py --num-transactions 500 --num-days 7

# Medium (dev)
python generate_synthetic_data.py --num-transactions 5000 --num-days 14

# Large (production-like)
python generate_synthetic_data.py --num-transactions 50000 --num-days 90
```

## Features

✅ All 10 transaction categories with proper labels  
✅ Vietnamese transaction messages  
✅ Realistic amount distributions by category  
✅ Proper MCC codes for merchants  
✅ Multi-day data for time-series features  
✅ Training/test split built-in  
✅ All foreign key relationships maintained  
✅ Privacy-preserving synthetic data  

## Verification

After generation, check:
```powershell
# Check file sizes
ls data/raw/*.parquet

# Quick inspection (requires pandas)
python -c "import pandas as pd; print(pd.read_parquet('data/raw/transactions_landing.parquet').info())"
```
