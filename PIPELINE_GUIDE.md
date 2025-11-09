# Transaction Classification Pipeline

## Quick Start

```bash
# Install GNU Make for Windows (if not installed)
# Download from: http://gnuwin32.sourceforge.net/packages/make.htm

# Run complete training pipeline
make pipeline-train

# See all commands
make help
```

## Prerequisites

1. **Install Dependencies**

   ```bash
   pip install pandas numpy torch scikit-learn scipy joblib fasttext-wheel tqdm
   ```

2. **Download FastText Model**
   - Place `cc.vi.300.bin` in `models/` directory
   - Download: <https://fasttext.cc/docs/en/crawl-vectors.html>

## Make Commands

### Complete Pipelines

```bash
make pipeline-train              # Full pipeline: data → encoders → features → classifier
make generate-data               # Generate synthetic data only
```

### Training Steps

```bash
make train-encoders              # Train context + sender encoders
make train-ctx-encoder           # Train context encoder only
make train-snd-encoder           # Train sender encoder only
make train-classifier            # Train tri-tower classifier
```

### Feature Building

```bash
make build-features-train        # Build all training features (includes labels)
make build-features-inference    # Build inference features (no labels)
make build-ctx-emb-train         # Context embeddings (training mode)
make build-ctx-emb-inference     # Context embeddings (inference mode)
make build-snd-features          # Sender features
make build-rcv-features          # Recipient features
```

### Utilities

```bash
make check-data                  # Check if data files exist
make check-models                # Check if models exist
make clean                       # Remove features/models (keep raw data)
make clean-all                   # Remove everything
```

## Pipeline Flow

```text
1. DATA → 2. ENCODERS → 3. FEATURES → 4. CLASSIFIER

generate-data
├── transactions_landing.parquet
├── recipient_entity.parquet
├── merchant_profile.parquet
└── person_profile.parquet
       ↓
train-encoders
├── train-ctx-encoder  → ctx_encoder.pt (128-dim)
└── train-snd-encoder  → snd_encoder.pt (128-dim)
       ↓
build-features-train
├── build-ctx-emb-train  → ctx_emb.parquet
├── build-snd-features   → x_snd/date=*/x_snd_user_state.parquet
└── build-rcv-features   → x_rcv/date=*/x_rcv_entity_state.parquet (96-dim)
       ↓
train-classifier
└── tri_tower_classifier.pt (352-dim input → 10 classes)
```

## Customization

### Modify Hyperparameters

Edit `Makefile` and change the arguments in respective targets:

- **Context Encoder**: `train-ctx-encoder` (--hidden, --emb-dim, --epochs)
- **Sender Encoder**: `train-snd-encoder` (--hidden, --maxlen, --epochs)
- **Classifier**: `train-classifier` (--hidden-dims, --dropout-rates, --epochs)

### Change Data Size

Edit `data/generate_synthetic_data.py`:

```python
NUM_TRANSACTIONS = 10000  # Change this
NUM_USERS = 1000
NUM_ENTITIES = 500
```

## Troubleshooting

**Missing modules**: `pip install <module-name>`

**FastText not found**: Download `cc.vi.300.bin` to `models/`

**Out of memory**: Reduce `--batch-size` in Makefile

**Missing labels in ctx_emb**: Use `make build-ctx-emb-train` not `build-ctx-emb-inference`

## Output Locations

- **Raw Data**: `data/raw/`
- **Features**: `data/feature_store/`
- **Models**: `models/`
- **Embeddings**: `data/processed/`
