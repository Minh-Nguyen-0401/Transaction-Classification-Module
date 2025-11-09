import pandas as pd
import numpy as np
import fasttext
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src._build_enc._encoders import CtxMLP, CtxAutoEncoder, CtxTrainer


def load_fasttext_model(ft_path):
    """Load pre-trained fastText model"""
    return fasttext.load_model(ft_path)


def prepare_features(df, ft_model):
    """Prepare numerical, categorical, and text features"""
    df = df.assign(
        amount_log=np.log1p(df["amount"].astype(float)),
        hour=pd.to_datetime(df["txn_time_utc"]).dt.hour.astype(int),
        hour_sin=np.sin(2*np.pi*pd.to_datetime(df["txn_time_utc"]).dt.hour/24.0),
        hour_cos=np.cos(2*np.pi*pd.to_datetime(df["txn_time_utc"]).dt.hour/24.0)
    )
    
    num = df[["amount_log","hour_sin","hour_cos"]].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    num = scaler.fit_transform(num)
    
    tt_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
    ch_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
    
    tt_oh = tt_ohe.fit_transform(df[["tranx_type"]].fillna("UNKNOWN"))
    ch_oh = ch_ohe.fit_transform(df[["channel"]].fillna("UNKNOWN"))
    
    txt = np.vstack([ft_model.get_sentence_vector(str(s)) for s in df["msg_content"].fillna("")]).astype(np.float32)
    
    X = np.concatenate([num, tt_oh, ch_oh, txt], axis=1)
    
    return X, scaler, tt_ohe, ch_ohe


def create_dataloader(X, batch_size=512):
    """Create dataloader from features"""
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def print_model_params(ctx_ae, enc):
    """Print parameter counts"""
    total_params = sum(p.numel() for p in ctx_ae.parameters() if p.requires_grad)
    params_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    params_dec = sum(p.numel() for p in ctx_ae.decoder.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params} (Encoder: {params_enc}, Decoder: {params_dec})")


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.src)
    ft_model = load_fasttext_model(args.ft_path)
    
    X, scaler, tt_ohe, ch_ohe = prepare_features(df, ft_model)
    dl = create_dataloader(X, batch_size=args.batch_size)
    
    enc = CtxMLP(d_in=X.shape[1], d_hidden=args.hidden, d_out=args.emb_dim)
    ctx_ae = CtxAutoEncoder(encoder=enc, d_hidden=args.hidden)
    
    print_model_params(ctx_ae, enc)
    
    trainer = CtxTrainer(ctx_ae, lr=args.lr)
    trainer.train(dl, epochs=args.epochs, verbose=True)
    
    torch.save(ctx_ae.state_dict(), outdir/"ctx_autoencoder.pt")
    torch.save(enc.state_dict(), outdir/"ctx_encoder.pt")
    
    joblib.dump(scaler, outdir/"ctx_num_scaler.pkl")
    joblib.dump(tt_ohe, outdir/"ctx_tt_ohe.pkl")
    joblib.dump(ch_ohe, outdir/"ctx_ch_ohe.pkl")
    
    print(f"\nModels saved to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Context Encoder with autoencoder")
    parser.add_argument("--src", type=str, 
                        default=str(ROOT / "data" / "raw" / "train_ctx.csv"),
                        help="Path to source CSV file")
    parser.add_argument("--ft-path", type=str, 
                        default=str(ROOT / "models" / "cc.vi.300.bin"),
                        help="Path to fastText model")
    parser.add_argument("--outdir", type=str, 
                        default=str(ROOT / "models"),
                        help="Output directory for models")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for training")
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden dimension for encoder/decoder")
    parser.add_argument("--emb-dim", type=int, default=128,
                        help="Embedding dimension (encoder output)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)
