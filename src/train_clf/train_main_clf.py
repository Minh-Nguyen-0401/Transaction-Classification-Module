import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import sys
import argparse
from datetime import datetime, timedelta
import logging

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.train_clf._main_clf import TriTowerClassifier, ClassifierTrainer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def load_feature_snapshots(feature_store_dir, feature_type, date_list):
    feature_dir = Path(feature_store_dir) / feature_type
    all_features = []
    
    for date_str in date_list:
        if feature_type == "x_snd":
            file_path = feature_dir / f"date={date_str}" / "x_snd_user_state.parquet"
        elif feature_type == "x_rcv":
            file_path = feature_dir / f"date={date_str}" / "x_rcv_entity_state.parquet"
        else:
            continue
        
        if file_path.exists():
            df_snapshot = pd.read_parquet(file_path)
            df_snapshot["_feature_date"] = date_str
            all_features.append(df_snapshot)
        else:
            print(f"Warning: {feature_type} not found for {date_str}")
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    else:
        return None


def load_and_merge_features(ctx_emb_path, feature_store_dir, txn_date_col, lookback_days=1):
    """Note: ctx_emb_path must be generated with --mode train"""
    df = pd.read_parquet(ctx_emb_path)
    
    required_cols = ["txn_id", "sender_id", "recipient_entity_id", txn_date_col, "label_l1"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Make sure to generate ctx_emb with --mode train flag."
        )
    
    df[txn_date_col] = pd.to_datetime(df[txn_date_col])
    df["_feature_date"] = (df[txn_date_col] - timedelta(days=lookback_days)).dt.strftime("%Y-%m-%d")
    
    min_feature_date = df["_feature_date"].min()
    max_feature_date = df["_feature_date"].max()
    print(f"Transaction date range: {df[txn_date_col].min().date()} to {df[txn_date_col].max().date()}")
    print(f"Feature date range needed: {min_feature_date} to {max_feature_date} (lookback={lookback_days} days)")
    
    ctx_cols = [c for c in df.columns if c.startswith("ctx_emb_")]
    print(f"Loaded context embeddings: {len(ctx_cols)} dimensions")
    ctx = df[["txn_id"] + ctx_cols]
    
    unique_dates = sorted(df["_feature_date"].unique())
    print(f"Loading {len(unique_dates)} daily snapshots for sender and recipient features")
    
    snd_all = load_feature_snapshots(feature_store_dir, "x_snd", unique_dates)
    if snd_all is not None:
        df = df.merge(snd_all, on=["sender_id", "_feature_date"], how="left")
        print(f"Merged sender features: {len(snd_all)} total embeddings across dates")
        snd = snd_all
    else:
        print("Warning: No sender features loaded")
        snd = None
    
    rcv_all = load_feature_snapshots(feature_store_dir, "x_rcv", unique_dates)
    if rcv_all is not None:
        rcv_all = rcv_all.rename(columns={"recipient_canon_id": "recipient_entity_id"})
        df = df.merge(rcv_all, on=["recipient_entity_id", "_feature_date"], how="left")
        print(f"Merged recipient features: {len(rcv_all)} total embeddings across dates")
        rcv = rcv_all
    else:
        print("Warning: No recipient features loaded")
        rcv = None
    
    return df, ctx, snd, rcv


def prepare_features(df, ctx, snd, rcv):
    cols_ctx = [c for c in ctx.columns if c.startswith("ctx_emb_")]
    cols_snd = [c for c in snd.columns if c.startswith("snd_emb_")] if snd is not None else []
    if cols_snd == []:
        print("Warning: No sender features found, proceeding without sender embeddings")
        raise
    cols_rcv = [c for c in rcv.columns if c.startswith("rcv_emb_")] if rcv is not None else []
    if cols_rcv == []:
        print("Warning: No recipient features found, proceeding without recipient embeddings")
        raise
    
    features = []
    if cols_ctx:
        features.append(df[cols_ctx].fillna(0).to_numpy(dtype=np.float32))
    if cols_snd:
        features.append(df[cols_snd].fillna(0).to_numpy(dtype=np.float32))
    if cols_rcv:
        features.append(df[cols_rcv].fillna(0).to_numpy(dtype=np.float32))
    
    X = np.concatenate(features, axis=1)
    
    print(f"Feature dimensions: CTX={len(cols_ctx)}, SND={len(cols_snd)}, RCV={len(cols_rcv)}, Total={X.shape[1]}")
    
    return X


def prepare_labels(df, label_map_path):
    import json
    
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    y2i = {key: value["index"] for key, value in label_map.items()}
    y = df["label_l1"].map(y2i).fillna(-1).astype(int).to_numpy()
    valid_mask = y >= 0
    
    print(f"Labels: {len(y2i)} classes from label map, {valid_mask.sum()} valid samples, {(~valid_mask).sum()} unknown labels")
    return y, y2i, valid_mask


def create_dataloaders(X, y, dates, batch_size=512, val_split=0.2, split_method='date'):
    """
    Create train/val dataloaders with date-based or random split.
    
    Args:
        X: Features array
        y: Labels array
        dates: Transaction dates (pd.Series or np.array)
        batch_size: Batch size for dataloaders
        val_split: Validation split ratio (0.2 = 20% validation)
        split_method: 'date' for temporal split, 'random' for stratified random split
    """
    if split_method == 'date':
        # Temporal split: last X% of dates for validation
        dates_sorted = pd.Series(dates).sort_values()
        split_date = dates_sorted.quantile(1 - val_split)
        
        train_mask = dates < split_date
        val_mask = dates >= split_date
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        train_dates = dates[train_mask]
        val_dates = dates[val_mask]
        
        logger.info(f"Date-based split:")
        logger.info(f"  Train period: {pd.Series(train_dates).min()} -> {pd.Series(train_dates).max()} ({len(X_train)} samples)")
        logger.info(f"  Val period: {pd.Series(val_dates).min()} -> {pd.Series(val_dates).max()} ({len(X_val)} samples)")
        
    elif split_method == 'random':
        # Random stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, stratify=y, random_state=42
        )
        logger.info(f"Random stratified split: Train={len(X_train)}, Val={len(X_val)}")
    else:
        raise ValueError(f"Unknown split_method: {split_method}. Use 'date' or 'random'")
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                           torch.tensor(y_val, dtype=torch.long))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading training data with lookback={args.lookback_days} days for features")
    print(f"Note: Make sure {args.ctx_emb_path} was generated with --mode train")
    
    df, ctx, snd, rcv = load_and_merge_features(
        args.ctx_emb_path,
        args.feature_store_dir,
        txn_date_col=args.txn_date_col,
        lookback_days=args.lookback_days
    )
    
    X = prepare_features(df, ctx, snd, rcv)
    y, y2i, valid_mask = prepare_labels(df, args.label_map_path)
    
    X = X[valid_mask]
    y = y[valid_mask]
    dates = df[args.txn_date_col].values[valid_mask]
    
    train_dl, val_dl = create_dataloaders(
        X, y, dates, 
        batch_size=args.batch_size, 
        val_split=args.val_split,
        split_method=args.split_method
    )
    
    model = TriTowerClassifier(
        d_in=X.shape[1], 
        n_classes=len(y2i),
        hidden_dims=args.hidden_dims,
        dropout_rates=args.dropout_rates
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    trainer = ClassifierTrainer(model, lr=args.lr)
    trainer.train(train_dl, val_dl, epochs=args.epochs, patience=args.patience, verbose=True)
    
    trainer.save_model(output_dir / "tri_tower_classifier.pt")
    joblib.dump(y2i, output_dir / "cls_label_map.pkl")
    
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tri-tower classifier")
    parser.add_argument("--ctx-emb-path", type=str,
                        default=str(ROOT / "data" / "processed" / "ctx_emb.parquet"),
                        help="Context embeddings parquet file (generated with --mode train)")
    parser.add_argument("--feature-store-dir", type=str,
                        default=str(ROOT / "data" / "feature_store"),
                        help="Feature store directory")
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "models"),
                        help="Output directory for models")
    parser.add_argument("--label-map-path", type=str,
                        default=str(ROOT / "data" / "label_map.json"),
                        help="Path to label mapping JSON file")
    parser.add_argument("--txn-date-col", type=str, default="txn_date",
                        help="Column name for transaction date in CSV")
    parser.add_argument("--lookback-days", type=int, default=1,
                        help="Days to lookback for features (to avoid data leak)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--split-method", type=str, default="date", choices=["date", "random"],
                        help="Split method: 'date' for temporal split (recommended), 'random' for stratified split")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256],
                        help="Hidden layer dimensions")
    parser.add_argument("--dropout-rates", type=float, nargs="+", default=[0.3, 0.2],
                        help="Dropout rates for each hidden layer")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of epochs")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience")
    
    args = parser.parse_args()
    main(args)
