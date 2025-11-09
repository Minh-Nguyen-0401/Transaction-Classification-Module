import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
import sys
import argparse
from datetime import datetime
import logging

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src._build_enc._encoders import SndEncoder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class SndFeatureGenerator:
    """Offline sender feature generator with sequential history"""
    
    def __init__(self, model_dir, lookback_days=10):
        self.model_dir = Path(model_dir)
        self.lookback_days = lookback_days
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load encoder and preprocessing artifacts"""
        self.tt_ohe = joblib.load(self.model_dir / "ctx_tt_ohe.pkl")
        self.ch_ohe = joblib.load(self.model_dir / "ctx_ch_ohe.pkl")
        
        n_tt_vocab = len(self.tt_ohe.categories_[0]) + 1
        n_ch_vocab = len(self.ch_ohe.categories_[0]) + 1
        
        self.encoder = SndEncoder(tt_vocab=n_tt_vocab, ch_vocab=n_ch_vocab, hidden=128)
        self.encoder.load_state_dict(torch.load(self.model_dir / "snd_encoder.pt", map_location="cpu"))
        self.encoder.eval()
    
    def _prepare_features(self, df):
        """Prepare transaction features"""
        df = df.assign(
            amount_log=np.log1p(df["amount"].astype(float)),
            hour=pd.to_datetime(df["txn_time_utc"]).dt.hour.astype(int),
            hour_sin=np.sin(2*np.pi*pd.to_datetime(df["txn_time_utc"]).dt.hour/24.0),
            hour_cos=np.cos(2*np.pi*pd.to_datetime(df["txn_time_utc"]).dt.hour/24.0)
        )
        
        tt_encoded = self.tt_ohe.transform(df[["tranx_type"]].fillna("UNKNOWN"))
        ch_encoded = self.ch_ohe.transform(df[["channel"]].fillna("UNKNOWN"))
        
        df["tt_idx"] = tt_encoded.argmax(axis=1) + 1
        df["ch_idx"] = ch_encoded.argmax(axis=1) + 1
        
        return df
    
    def generate_embeddings(self, df, snapshot_date):
        """Generate embeddings for senders active on snapshot_date using lookback window"""
        df["txn_date"] = pd.to_datetime(df["txn_time_utc"]).dt.date
        snapshot_date = pd.to_datetime(snapshot_date).date()
        
        # Find senders active on snapshot_date
        active_senders = df[df["txn_date"] == snapshot_date]["sender_id"].unique()
        logger.info(f"Found {len(active_senders)} active senders on {snapshot_date}")
        
        lookback_start = snapshot_date - pd.Timedelta(days=self.lookback_days)
        
        df_filtered = df[
            (df["sender_id"].isin(active_senders)) &
            (df["txn_date"] >= lookback_start) &
            (df["txn_date"] <= snapshot_date)
        ].copy()
        
        logger.info(f"Processing {len(df_filtered)} transactions (lookback: {self.lookback_days} days)")
        
        df_filtered = self._prepare_features(df_filtered.sort_values("txn_time_utc"))
        
        embeddings = []
        sender_ids = []
        
        for sender_id in active_senders:
            group = df_filtered[df_filtered["sender_id"] == sender_id]
            
            if len(group) == 0:
                continue
            
            a = group[["amount_log","hour_sin","hour_cos"]].to_numpy(dtype=np.float32)
            t = group["tt_idx"].to_numpy(dtype=np.int64)
            c = group["ch_idx"].to_numpy(dtype=np.int64)
            
            x_num = torch.tensor(a).unsqueeze(0)
            x_tt = torch.tensor(t).unsqueeze(0)
            x_ch = torch.tensor(c).unsqueeze(0)
            
            with torch.no_grad():
                h = self.encoder(x_num, x_tt, x_ch).squeeze(0).numpy()
            
            embeddings.append(h)
            sender_ids.append(sender_id)
        
        E = np.vstack(embeddings)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        
        return sender_ids, E
    
    def _get_dates_to_process(self, df, output_dir, snapshot_date):
        """Determine which dates to process based on existing snapshots"""
        output_dir = Path(output_dir)
        
        if snapshot_date is not None:
            return [snapshot_date]
        
        df["txn_date"] = pd.to_datetime(df["txn_time_utc"]).dt.date
        all_dates = sorted(df["txn_date"].unique())
        all_dates_str = [d.strftime("%Y-%m-%d") for d in all_dates]
        
        existing_dates = set()
        for date_str in all_dates_str:
            snapshot_path = output_dir / f"date={date_str}" / "x_snd_user_state.parquet"
            if snapshot_path.exists():
                existing_dates.add(date_str)
        
        dates_to_process = [d for d in all_dates_str if d not in existing_dates]
        
        logger.info(f"Found {len(all_dates_str)} unique transaction dates")
        logger.info(f"Existing snapshots: {len(existing_dates)} dates")
        logger.info(f"To process: {len(dates_to_process)} dates")
        
        if dates_to_process:
            logger.info(f"Processing dates: {dates_to_process[:5]}{'...' if len(dates_to_process) > 5 else ''}")
        
        return dates_to_process
    
    def process_and_save(self, input_path, output_dir, snapshot_date=None):
        """Process transactions and save to feature store"""
        df = pd.read_parquet(input_path)
        
        dates_to_process = self._get_dates_to_process(df, output_dir, snapshot_date)
        
        if not dates_to_process:
            logger.info("No new dates to process. All snapshots already exist.")
            return None
        
        all_feature_dfs = []
        for date_str in dates_to_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing snapshot date: {date_str}")
            logger.info(f"{'='*60}")
            
            sender_ids, embeddings = self.generate_embeddings(df, date_str)
            
            feature_df = pd.DataFrame(embeddings, columns=[f"snd_emb_{i+1}" for i in range(embeddings.shape[1])])
            feature_df.insert(0, "sender_id", sender_ids)
            feature_df["feature_version"] = "v1"
            feature_df["snapshot_date"] = date_str
            feature_df["updated_at"] = datetime.now().isoformat()
            
            # Save daily snapshot
            daily_path = Path(output_dir) / f"date={date_str}" / "x_snd_user_state.parquet"
            daily_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_parquet(daily_path, index=False)
            logger.info(f"Generated daily snapshot for {len(sender_ids)} senders -> {daily_path}")
            
            all_feature_dfs.append(feature_df)
        
        # Upsert all processed dates to cumulative feature store
        cumulative_path = Path(output_dir) / "date=ALL" / "x_snd_user_state.parquet"
        cumulative_path.parent.mkdir(parents=True, exist_ok=True)
        
        combined_new_df = pd.concat(all_feature_dfs, ignore_index=True)
        
        if cumulative_path.exists():
            # Load existing cumulative data
            existing_df = pd.read_parquet(cumulative_path)
            
            # Remove old records for updated senders
            all_updated_senders = combined_new_df["sender_id"].unique()
            existing_df = existing_df[~existing_df["sender_id"].isin(all_updated_senders)]
            updated_df = pd.concat([existing_df, combined_new_df], ignore_index=True)
            
            logger.info(f"Updated cumulative store: {len(existing_df)} existing + {len(combined_new_df)} new/updated = {len(updated_df)} total")
        else:
            updated_df = combined_new_df
            logger.info(f"Created new cumulative store with {len(combined_new_df)} senders")
        
        updated_df.to_parquet(cumulative_path, index=False)
        logger.info(f"Cumulative store saved -> {cumulative_path}")
        
        return combined_new_df


def main(args):
    generator = SndFeatureGenerator(args.model_dir, lookback_days=args.lookback_days)
    generator.process_and_save(args.input, args.output_dir, args.snapshot_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sender embeddings from transaction history (offline)")
    parser.add_argument("--input", type=str, 
                        default=str(ROOT / "data" / "raw" / "transactions_landing.parquet"),
                        help="Input parquet file with transactions")
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "data" / "feature_store" / "x_snd"),
                        help="Output directory for feature store")
    parser.add_argument("--model-dir", type=str,
                        default=str(ROOT / "models"),
                        help="Directory containing model artifacts")
    parser.add_argument("--snapshot-date", type=str, default=None,
                        help="Snapshot date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--lookback-days", type=int, default=10,
                        help="Number of days to look back for transaction history")
    
    args = parser.parse_args()
    main(args)
