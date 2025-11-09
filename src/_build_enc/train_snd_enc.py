import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib
from pathlib import Path
import sys
import argparse
import logging

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src._build_enc._encoders import SndEncoder, SndTrainer
from src._build_enc._seq import SeqDS, collate_seq

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def load_and_prepare_data(src_path, tt_ohe, ch_ohe):
    """Load data and perform feature engineering"""
    df = pd.read_parquet(src_path).sort_values("txn_time_utc")
    
    df = df.assign(
        amount_log=np.log1p(df["amount"].astype(float)),
        hour=pd.to_datetime(df["txn_time_utc"]).dt.hour.astype(int),
        hour_sin=np.sin(2*np.pi*pd.to_datetime(df["txn_time_utc"]).dt.hour/24.0),
        hour_cos=np.cos(2*np.pi*pd.to_datetime(df["txn_time_utc"]).dt.hour/24.0)
    )
    
    tt_encoded = tt_ohe.transform(df[["tranx_type"]].fillna("UNKNOWN"))
    ch_encoded = ch_ohe.transform(df[["channel"]].fillna("UNKNOWN"))
    
    df["tt_idx"] = tt_encoded.argmax(axis=1) + 1
    df["ch_idx"] = ch_encoded.argmax(axis=1) + 1
    
    return df


def create_amount_bins(df):
    """Bin amounts using quantiles"""
    q = df["amount_log"].quantile([0.2, 0.4, 0.6, 0.8]).values
    
    def bin_amount(x):
        if x <= q[0]: return 0
        if x <= q[1]: return 1
        if x <= q[2]: return 2
        if x <= q[3]: return 3
        return 4
    
    df["amt_bin"] = df["amount_log"].apply(bin_amount).astype(int)
    return df, q


def create_dataloader(df, batch_size=256, maxlen=100):
    """Create sequential dataset and dataloader"""
    g = df.groupby("sender_id", sort=False)
    ds = SeqDS(g, maxlen=maxlen)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_seq, num_workers=0)
    return dl


def print_model_params(enc, trainer):
    total_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    head_t_params = sum(p.numel() for p in trainer.head_tranx.parameters() if p.requires_grad)
    head_a_params = sum(p.numel() for p in trainer.head_amount.parameters() if p.requires_grad)
    
    logger.info(f"Total trainable parameters: {total_params + head_t_params + head_a_params}")
    logger.info(f"Encoder: {total_params}, Head_Tranx: {head_t_params}, Head_Amount: {head_a_params}")


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    tt_ohe = joblib.load(outdir/"ctx_tt_ohe.pkl")
    ch_ohe = joblib.load(outdir/"ctx_ch_ohe.pkl")
    
    df = load_and_prepare_data(args.src, tt_ohe, ch_ohe)
    df, q = create_amount_bins(df)
    dl = create_dataloader(df, batch_size=args.batch_size, maxlen=args.maxlen)
    
    n_tt_vocab = len(tt_ohe.categories_[0]) + 1
    n_ch_vocab = len(ch_ohe.categories_[0]) + 1
    
    enc = SndEncoder(tt_vocab=n_tt_vocab, ch_vocab=n_ch_vocab, hidden=args.hidden)
    trainer = SndTrainer(encoder=enc, n_tranx_types=n_tt_vocab, n_amount_bins=5, lr=args.lr)
    
    print_model_params(enc, trainer)
    
    trainer.train(dl, epochs=args.epochs, verbose=True)
    
    trainer.save_encoder(outdir/"snd_encoder.pt")
    joblib.dump(q, outdir/"snd_amount_quantiles.pkl")
    
    logger.info(f"Models saved to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sender Encoder with multi-task learning")
    parser.add_argument("--src", type=str, 
                        default=str(ROOT / "data" / "raw" / "transactions_landing.parquet"),
                        help="Path to source parquet file")
    parser.add_argument("--outdir", type=str, 
                        default=str(ROOT / "models"),
                        help="Output directory for models")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--maxlen", type=int, default=100,
                        help="Maximum sequence length")
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden dimension for encoder")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)

