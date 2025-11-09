# DVC (Data Version Control) Guide

## What is DVC?

DVC is a version control system for data and models. It works alongside Git to track large files, datasets, and ML models without storing them in Git.

## Installation

```bash
pip install dvc
pip install dvc[s3]      # For AWS S3
pip install dvc[gdrive]  # For Google Drive
pip install dvc[azure]   # For Azure
```

## Initial Setup

### 1. Initialize DVC in Your Repository

```bash
# Initialize DVC (run this once)
dvc init

# This creates:
# - .dvc/ directory (DVC configuration)
# - .dvcignore file
```

### 2. Configure Remote Storage

DVC supports multiple storage backends:

#### Option A: Local Remote (for testing)

```bash
# Create a local storage location
mkdir D:\dvc-storage\tranx_clf

# Add as DVC remote
dvc remote add -d myremote D:\dvc-storage\tranx_clf
```

#### Option B: Google Drive

```bash
# Add Google Drive as remote
dvc remote add -d gdrive gdrive://your-folder-id

# Authenticate (will open browser)
dvc remote modify gdrive gdrive_acknowledge_abuse true
```

#### Option C: AWS S3

```bash
# Add S3 bucket as remote
dvc remote add -d myremote s3://my-bucket/tranx-clf-data

# Configure credentials
dvc remote modify myremote access_key_id 'your-access-key'
dvc remote modify myremote secret_access_key 'your-secret-key'
```

#### Option D: Azure Blob Storage

```bash
# Add Azure as remote
dvc remote add -d myremote azure://mycontainer/path

# Configure credentials
dvc remote modify myremote account_name 'myaccount'
dvc remote modify myremote account_key 'mykey'
```

## Tracking Data and Models

### Track Raw Data

```bash
# Track entire raw data directory
dvc add data/raw

# This creates:
# - data/raw.dvc (metadata file to commit to Git)
# - data/raw/.gitignore (Git will ignore actual data)
```

### Track Processed Data

```bash
dvc add data/processed
dvc add data/feature_store
```

### Track Models

```bash
# Track individual model files
dvc add models/ctx_encoder.pt
dvc add models/snd_encoder.pt
dvc add models/tri_tower_classifier.pt

# Or track entire models directory
dvc add models
```

### Commit DVC Files to Git

```bash
# Add .dvc files to Git
git add data/raw.dvc data/processed.dvc models.dvc
git add .dvc/.gitignore .dvc/config

# Commit
git commit -m "Track data and models with DVC"
```

## Pushing and Pulling Data

### Push Data to Remote

```bash
# Push all tracked data/models to remote storage
dvc push

# Push specific file
dvc push data/raw.dvc
```

### Pull Data from Remote

```bash
# Pull all tracked data/models from remote storage
dvc pull

# Pull specific file
dvc pull data/raw.dvc
```

## Pipeline with DVC

### Define Pipeline Stages

DVC can track your entire ML pipeline:

```bash
# Stage 1: Generate data
dvc stage add -n generate_data \
    -d data/generate_synthetic_data.py \
    -o data/raw/transactions_landing.parquet \
    -o data/raw/recipient_entity.parquet \
    -o data/raw/merchant_profile.parquet \
    -o data/raw/person_profile.parquet \
    python data/generate_synthetic_data.py --num-transactions 100000

# Stage 2: Train context encoder
dvc stage add -n train_ctx_encoder \
    -d src/_build_enc/train_ctx_enc.py \
    -d data/raw/transactions_landing.parquet \
    -o models/ctx_encoder.pt \
    -o models/ctx_num_scaler.pkl \
    python src/_build_enc/train_ctx_enc.py

# Stage 3: Train sender encoder
dvc stage add -n train_snd_encoder \
    -d src/_build_enc/train_snd_enc.py \
    -d data/raw/transactions_landing.parquet \
    -p models/ctx_tt_ohe.pkl \
    -o models/snd_encoder.pt \
    python src/_build_enc/train_snd_enc.py

# Stage 4: Build context embeddings
dvc stage add -n build_ctx_embeddings \
    -d src/preprocess/_x_ctx_online.py \
    -d data/raw/transactions_landing.parquet \
    -d models/ctx_encoder.pt \
    -o data/processed/ctx_emb.parquet \
    python src/preprocess/_x_ctx_online.py --mode train

# Stage 5: Build sender features
dvc stage add -n build_snd_features \
    -d src/preprocess/_x_snd_offline.py \
    -d data/raw/transactions_landing.parquet \
    -d models/snd_encoder.pt \
    -o data/feature_store/x_snd \
    python src/preprocess/_x_snd_offline.py

# Stage 6: Build recipient features
dvc stage add -n build_rcv_features \
    -d src/preprocess/_x_rcv_offline.py \
    -d data/raw \
    -o data/feature_store/x_rcv \
    python src/preprocess/_x_rcv_offline.py

# Stage 7: Train classifier
dvc stage add -n train_classifier \
    -d src/train_clf/train_main_clf.py \
    -d data/processed/ctx_emb.parquet \
    -d data/feature_store/x_snd \
    -d data/feature_store/x_rcv \
    -o models/tri_tower_classifier.pt \
    python src/train_clf/train_main_clf.py
```

### Run Pipeline

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train_classifier

# Force re-run even if no changes
dvc repro -f
```

### Visualize Pipeline

```bash
# Show pipeline DAG
dvc dag

# Show pipeline metrics
dvc metrics show

# Generate pipeline visualization
dvc dag --md > pipeline.md
```

## Practical Workflow

### Complete Setup

```bash
# 1. Initialize DVC
dvc init
git add .dvc
git commit -m "Initialize DVC"

# 2. Configure remote storage
dvc remote add -d myremote D:\dvc-storage\tranx_clf
git add .dvc/config
git commit -m "Configure DVC remote"

# 3. Track data and models
dvc add data/raw
dvc add data/processed
dvc add data/feature_store
dvc add models

# 4. Commit DVC files
git add data/raw.dvc data/processed.dvc data/feature_store.dvc models.dvc
git add data/.gitignore models/.gitignore
git commit -m "Track data and models with DVC"

# 5. Push to DVC remote
dvc push

# 6. Push to Git remote
git push
```

### Collaborator Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd tranx_clf

# 2. Pull data from DVC remote
dvc pull

# Now all data and models are available!
```

### Update Data/Models

```bash
# 1. Make changes (run pipeline, update data, etc.)
make pipeline-train

# 2. Update DVC tracking
dvc add data/raw
dvc add models

# 3. Commit changes
git add data/raw.dvc models.dvc
git commit -m "Update models with new training run"

# 4. Push to DVC remote
dvc push

# 5. Push to Git
git push
```

### Switch Between Versions

```bash
# Checkout specific Git commit
git checkout <commit-hash>

# Pull corresponding data/models
dvc pull

# Your workspace now has data/models from that commit!
```

## Best Practices

### 1. Track Large Files Only with DVC

- Raw data files (`.parquet`, `.csv`)
- Trained models (`.pt`, `.pkl`, `.h5`)
- Feature stores
- Large binary files

### 2. Keep in Git

- Source code (`.py`)
- Configuration files (`.json`, `.yaml`)
- DVC files (`.dvc`)
- Documentation (`.md`)
- Requirements (`requirements.txt`)

### 3. Use .gitignore and .dvcignore

- `.gitignore`: Prevent Git from tracking large files
- `.dvcignore`: Prevent DVC from scanning unnecessary files

### 4. Commit DVC Files After Changes

```bash
# Always commit .dvc files after dvc add
git add data/raw.dvc
git commit -m "Update raw data"
```

### 5. Push to Both Remotes

```bash
# Push data to DVC remote
dvc push

# Push metadata to Git remote
git push
```

## Troubleshooting

### Issue: "dvc push" fails

```bash
# Check remote configuration
dvc remote list

# Check remote connection
dvc remote default

# Reconfigure if needed
dvc remote modify myremote url <new-url>
```

### Issue: "dvc pull" fails

```bash
# Check if remote is configured
dvc remote list

# Try pulling specific file
dvc pull data/raw.dvc -v

# Check cache
dvc cache dir
```

### Issue: Large files in Git

```bash
# If you accidentally committed large files to Git:
git rm --cached data/raw/*.parquet
dvc add data/raw
git add data/raw.dvc data/.gitignore
git commit -m "Move large files to DVC"
```

## Quick Reference

```bash
# Initialize
dvc init

# Configure remote
dvc remote add -d myremote <storage-url>

# Track files
dvc add <file-or-directory>

# Push/Pull
dvc push                    # Push all
dvc pull                    # Pull all
dvc push <file>.dvc        # Push specific
dvc pull <file>.dvc        # Pull specific

# Pipeline
dvc stage add -n <name> -d <deps> -o <outputs> <command>
dvc repro                  # Run pipeline
dvc dag                    # Visualize pipeline

# Status
dvc status                 # Check DVC status
dvc diff                   # Show changes

# Remove
dvc remove <file>.dvc      # Untrack file
```

## Example: Complete Workflow for This Project

```bash
# Initial setup
cd tranx_clf
dvc init
dvc remote add -d local D:\dvc-storage\tranx_clf

# Track data and models
dvc add data/raw
dvc add data/processed
dvc add data/feature_store
dvc add models

# Commit to Git
git add .dvc data/*.dvc models.dvc data/.gitignore models/.gitignore
git commit -m "Initialize DVC and track data/models"

# Push to DVC remote
dvc push

# Push to Git remote
git push origin main

# When someone else clones:
# git clone <repo>
# dvc pull
# Now they have all data and models!
```
