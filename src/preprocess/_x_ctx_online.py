import pandas as pd
import numpy as np
import fasttext
import torch
import joblib
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src._build_enc._encoders import CtxMLP


class CtxEmbedder:
    """Online context embedder for transaction inference"""
    
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all required artifacts"""
        self.scaler = joblib.load(self.model_dir / "ctx_num_scaler.pkl")
        self.tt_ohe = joblib.load(self.model_dir / "ctx_tt_ohe.pkl")
        self.ch_ohe = joblib.load(self.model_dir / "ctx_ch_ohe.pkl")
        
        ft_path = self.model_dir / "cc.vi.300.bin"
        self.ft = fasttext.load_model(str(ft_path))
        
        state = torch.load(self.model_dir / "ctx_encoder.pt", map_location="cpu")
        in_dim = list(state.values())[0].shape[1]
        self.encoder = CtxMLP(d_in=in_dim, d_hidden=256, d_out=128)
        self.encoder.load_state_dict(state)
        self.encoder.eval()
    
    def _prepare_features(self, df):
        """Prepare numerical, categorical, and text features"""
        df = df.assign(
            amount_log=np.log1p(df["amount"].astype(float)),
            hour=pd.to_datetime(df["txn_time_utc"]).dt.hour.astype(int),
            hour_sin=np.sin(2*np.pi*pd.to_datetime(df["txn_time_utc"]).dt.hour/24.0),
            hour_cos=np.cos(2*np.pi*pd.to_datetime(df["txn_time_utc"]).dt.hour/24.0)
        )
        
        num = df[["amount_log","hour_sin","hour_cos"]].to_numpy(dtype=np.float32)
        num = self.scaler.transform(num)
        
        tt_oh = self.tt_ohe.transform(df[["tranx_type"]].fillna("UNKNOWN"))
        ch_oh = self.ch_ohe.transform(df[["channel"]].fillna("UNKNOWN"))
        
        txt = np.vstack([self.ft.get_sentence_vector(str(s)) 
                        for s in df["msg_content"].fillna("")]).astype(np.float32)
        
        X = np.concatenate([num, tt_oh, ch_oh, txt], axis=1)
        return X
    
    def embed_batch(self, df):
        """Generate embeddings for a batch of transactions"""
        X = self._prepare_features(df)
        
        with torch.no_grad():
            h = self.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
        
        return h
    
    def process_file(self, input_path, out_parquet):
        """Process input file (CSV or Parquet) and save embeddings to parquet"""
        input_path = Path(input_path)
        
        # Auto-detect file type
        if input_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(input_path)
        elif input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}. Use .csv or .parquet")
        
        embeddings = self.embed_batch(df)
        
        out = pd.DataFrame(embeddings, columns=[f"ctx_emb_{i+1}" for i in range(embeddings.shape[1])])
        out.insert(0, "txn_id", df["txn_id"].astype(str))
        
        out_path = Path(out_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        
        print(f"Processed {len(df)} transactions → {out_path}")
        return out


def main(args):
    embedder = CtxEmbedder(args.model_dir)
    embedder.process_file(args.input, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate context embeddings for transactions (online inference)")
    parser.add_argument("--input", type=str,
                        default=str(ROOT / "data" / "raw" / "transactions_landing.parquet"),
                        help="Input file path (CSV or Parquet)")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "data" / "processed" / "ctx_emb.parquet"),
                        help="Output parquet file path")
    parser.add_argument("--model-dir", type=str, 
                        default=str(ROOT / "models"),
                        help="Directory containing model artifacts")
    
    args = parser.parse_args()
    main(args)
