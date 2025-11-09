# Makefile for Transaction Classification Pipeline
# For Windows PowerShell

PYTHON = python -u
DATA_DIR = data
RAW_DIR = $(DATA_DIR)\raw
PROCESSED_DIR = $(DATA_DIR)\processed
FEATURE_STORE = $(DATA_DIR)\feature_store
MODELS_DIR = models
LOGS_DIR = logs

# Data files
TRANSACTIONS = $(RAW_DIR)\transactions_landing.parquet
RECIPIENTS = $(RAW_DIR)\recipient_entity.parquet
MERCHANTS = $(RAW_DIR)\merchant_profile.parquet
PERSONS = $(RAW_DIR)\person_profile.parquet

# Model artifacts
FASTTEXT_MODEL = $(MODELS_DIR)\cc.vi.300.bin
CTX_ENCODER = $(MODELS_DIR)\ctx_encoder.pt
SND_ENCODER = $(MODELS_DIR)\snd_encoder.pt
CLASSIFIER = $(MODELS_DIR)\tri_tower_classifier.pt

# Feature outputs
CTX_EMB = $(PROCESSED_DIR)\ctx_emb.parquet
X_SND_FEATURES = $(FEATURE_STORE)\x_snd
X_RCV_FEATURES = $(FEATURE_STORE)\x_rcv

.PHONY: all clean help generate-data train-encoders train-classifier pipeline-train pipeline-inference setup-logs

# Default target
all: help

help:
	@echo "Transaction Classification Pipeline - Make Targets"
	@echo "=================================================="
	@echo ""
	@echo "Data Generation:"
	@echo "  make generate-data          Generate synthetic data for all tables"
	@echo ""
	@echo "Training Pipeline:"
	@echo "  make train-encoders         Train context and sender encoders"
	@echo "  make build-features-train   Build training features (ctx, snd, rcv)"
	@echo "  make train-classifier       Train the tri-tower classifier"
	@echo "  make pipeline-train         Run complete training pipeline"
	@echo ""
	@echo "Inference Pipeline:"
	@echo "  make build-features-inference   Build inference features"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make train-ctx-encoder      Train context encoder only"
	@echo "  make train-snd-encoder      Train sender encoder only"
	@echo "  make build-ctx-emb-train    Build context embeddings (training mode)"
	@echo "  make build-ctx-emb-inference Build context embeddings (inference mode)"
	@echo "  make build-snd-features     Build sender features"
	@echo "  make build-rcv-features     Build recipient features"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean                  Clean generated features and models"
	@echo "  make clean-all              Clean everything including raw data"
	@echo "  make setup-logs             Create log directories"

# ==================== SETUP ====================

setup-logs:
	@if not exist $(LOGS_DIR) mkdir $(LOGS_DIR)
	@if not exist $(LOGS_DIR)\data mkdir $(LOGS_DIR)\data
	@if not exist $(LOGS_DIR)\encoders mkdir $(LOGS_DIR)\encoders
	@if not exist $(LOGS_DIR)\features mkdir $(LOGS_DIR)\features
	@if not exist $(LOGS_DIR)\classifier mkdir $(LOGS_DIR)\classifier

# ==================== DATA GENERATION ====================

generate-data: setup-logs
	@echo "Generating synthetic data..."
	@echo "Logging to: $(LOGS_DIR)\data\generate_data.log"
	$(PYTHON) data\generate_synthetic_data.py --num-transactions 10000 --num-days 30 --start-date 2025-09-01 > $(LOGS_DIR)\data\generate_data.log 2>&1

# ==================== ENCODER TRAINING ====================

train-ctx-encoder: $(TRANSACTIONS) setup-logs
	@echo "Training context encoder..."
	@echo "Logging to: $(LOGS_DIR)\encoders\train_ctx_encoder.log"
	$(PYTHON) src\_build_enc\train_ctx_enc.py --src $(TRANSACTIONS) --ft-path $(FASTTEXT_MODEL) --outdir $(MODELS_DIR) --hidden 256 --emb-dim 128 --batch-size 512 --lr 1e-3 --epochs 5 > $(LOGS_DIR)\encoders\train_ctx_encoder.log 2>&1

train-snd-encoder: $(TRANSACTIONS) train-ctx-encoder setup-logs
	@echo "Training sender encoder..."
	@echo "Logging to: $(LOGS_DIR)\encoders\train_snd_encoder.log"
	$(PYTHON) src\_build_enc\train_snd_enc.py --src $(TRANSACTIONS) --outdir $(MODELS_DIR) --batch-size 256 --maxlen 100 --hidden 128 --lr 1e-3 --epochs 5 > $(LOGS_DIR)\encoders\train_snd_encoder.log 2>&1

train-encoders: train-ctx-encoder train-snd-encoder
	@echo "All encoders trained successfully!"

# ==================== FEATURE BUILDING (TRAINING MODE) ====================

build-ctx-emb-train: $(TRANSACTIONS) $(CTX_ENCODER) setup-logs
	@echo "Building context embeddings (training mode)..."
	@echo "Logging to: $(LOGS_DIR)\features\build_ctx_emb_train.log"
	$(PYTHON) src/preprocess/_x_ctx_online.py \
		--input $(TRANSACTIONS) \
		--output $(CTX_EMB) \
		--model-dir $(MODELS_DIR) \
		--mode train > $(LOGS_DIR)\features\build_ctx_emb_train.log 2>&1

build-ctx-emb-inference: $(TRANSACTIONS) $(CTX_ENCODER) setup-logs
	@echo "Building context embeddings (inference mode)..."
	@echo "Logging to: $(LOGS_DIR)\features\build_ctx_emb_inference.log"
	$(PYTHON) src/preprocess/_x_ctx_online.py \
		--input $(TRANSACTIONS) \
		--output $(CTX_EMB) \
		--model-dir $(MODELS_DIR) \
		--mode inference > $(LOGS_DIR)\features\build_ctx_emb_inference.log 2>&1

build-snd-features: $(TRANSACTIONS) $(SND_ENCODER) setup-logs
	@echo "Building sender features..."
	@echo "Logging to: $(LOGS_DIR)\features\build_snd_features.log"
	$(PYTHON) src/preprocess/_x_snd_offline.py \
		--input $(TRANSACTIONS) \
		--output-dir $(X_SND_FEATURES) \
		--model-dir $(MODELS_DIR) \
		--lookback-days 10 > $(LOGS_DIR)\features\build_snd_features.log 2>&1

build-rcv-features: $(TRANSACTIONS) $(RECIPIENTS) $(MERCHANTS) $(PERSONS) setup-logs
	@echo "Building recipient features..."
	@echo "Logging to: $(LOGS_DIR)\features\build_rcv_features.log"
	$(PYTHON) src/preprocess/_x_rcv_offline.py \
		--data-dir $(RAW_DIR) \
		--output-dir $(X_RCV_FEATURES) \
		--svd-dim 64 \
		--hash-dim 32 \
		--lookback-days 10 > $(LOGS_DIR)\features\build_rcv_features.log 2>&1

build-features-train: build-ctx-emb-train build-snd-features build-rcv-features
	@echo "All training features built successfully!"

build-features-inference: build-ctx-emb-inference build-snd-features build-rcv-features
	@echo "All inference features built successfully!"

# ==================== CLASSIFIER TRAINING ====================

train-classifier: $(CTX_EMB) setup-logs
	@echo "Training tri-tower classifier..."
	@echo "Logging to: $(LOGS_DIR)\classifier\train_classifier.log"
	$(PYTHON) src/train_clf/train_main_clf.py \
		--ctx-emb-path $(CTX_EMB) \
		--feature-store-dir $(FEATURE_STORE) \
		--output-dir $(MODELS_DIR) \
		--label-map-path $(DATA_DIR)/label_map.json \
		--txn-date-col txn_date \
		--lookback-days 1 \
		--batch-size 512 \
		--val-split 0.2 \
		--hidden-dims 512 256 \
		--dropout-rates 0.3 0.2 \
		--lr 1e-3 \
		--epochs 8 \
		--patience 3 > $(LOGS_DIR)\classifier\train_classifier.log 2>&1

# ==================== COMPLETE PIPELINES ====================

pipeline-train: generate-data train-encoders build-features-train train-classifier
	@echo ""
	@echo "========================================"
	@echo "Training pipeline completed successfully!"
	@echo "========================================"
	@echo ""
	@echo "Models saved in: $(MODELS_DIR)"
	@echo "Features saved in: $(FEATURE_STORE)"
	@echo ""

pipeline-inference: build-features-inference
	@echo ""
	@echo "=========================================="
	@echo "Inference features built successfully!"
	@echo "=========================================="
	@echo ""
	@echo "Features saved in: $(FEATURE_STORE)"
	@echo ""

# ==================== CLEANING ====================

clean:
	@echo "Cleaning generated features and models..."
	@if exist "$(PROCESSED_DIR)" rmdir /s /q "$(PROCESSED_DIR)"
	@if exist "$(FEATURE_STORE)" rmdir /s /q "$(FEATURE_STORE)"
	@if exist "$(MODELS_DIR)\ctx_encoder.pt" del /q "$(MODELS_DIR)\ctx_encoder.pt"
	@if exist "$(MODELS_DIR)\ctx_autoencoder.pt" del /q "$(MODELS_DIR)\ctx_autoencoder.pt"
	@if exist "$(MODELS_DIR)\snd_encoder.pt" del /q "$(MODELS_DIR)\snd_encoder.pt"
	@if exist "$(MODELS_DIR)\tri_tower_classifier.pt" del /q "$(MODELS_DIR)\tri_tower_classifier.pt"
	@if exist "$(MODELS_DIR)\*.pkl" del /q "$(MODELS_DIR)\*.pkl"
	@if exist "$(LOGS_DIR)" rmdir /s /q "$(LOGS_DIR)"
	@echo "Clean completed!"

clean-all: clean
	@echo "Cleaning all data including raw files..."
	@if exist "$(RAW_DIR)" rmdir /s /q "$(RAW_DIR)"
	@if exist "$(MODELS_DIR)" rmdir /s /q "$(MODELS_DIR)"
	@echo "Clean-all completed!"

# ==================== QUICK COMMANDS ====================

# Quick test with small dataset
test-pipeline: setup-logs
	@echo "Running quick test pipeline..."
	@echo "Logging to: $(LOGS_DIR)\data\test_generate_data.log"
	$(PYTHON) data/generate_synthetic_data.py --num-transactions 1000 > $(LOGS_DIR)\data\test_generate_data.log 2>&1
	@$(MAKE) train-encoders
	@$(MAKE) build-features-train
	@$(MAKE) train-classifier

# Check if all required files exist
check-data:
	@echo "Checking data files..."
	@if exist $(TRANSACTIONS) (echo [OK] Transactions file exists) else (echo [MISSING] $(TRANSACTIONS))
	@if exist $(RECIPIENTS) (echo [OK] Recipients file exists) else (echo [MISSING] $(RECIPIENTS))
	@if exist $(MERCHANTS) (echo [OK] Merchants file exists) else (echo [MISSING] $(MERCHANTS))
	@if exist $(PERSONS) (echo [OK] Persons file exists) else (echo [MISSING] $(PERSONS))
	@if exist $(FASTTEXT_MODEL) (echo [OK] FastText model exists) else (echo [MISSING] $(FASTTEXT_MODEL))

check-models:
	@echo "Checking trained models..."
	@if exist $(CTX_ENCODER) (echo [OK] Context encoder exists) else (echo [MISSING] $(CTX_ENCODER))
	@if exist $(SND_ENCODER) (echo [OK] Sender encoder exists) else (echo [MISSING] $(SND_ENCODER))
	@if exist $(CLASSIFIER) (echo [OK] Classifier exists) else (echo [MISSING] $(CLASSIFIER))
