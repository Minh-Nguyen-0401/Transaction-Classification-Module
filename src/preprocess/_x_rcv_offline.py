import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction import FeatureHasher
import argparse
from datetime import datetime
import sys
import logging

ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class RecipientFeatureGenerator:
    """Offline recipient feature generator using SVD and metadata hashing"""
    
    def __init__(self, svd_dim=64, hash_dim=32, lookback_days=10):
        self.svd_dim = svd_dim
        self.hash_dim = hash_dim
        self.lookback_days = lookback_days
        self.hasher = FeatureHasher(n_features=hash_dim, input_type="string")
    
    def _build_interaction_matrix(self, tx_df, snapshot_date):
        """Build sender-recipient interaction matrix for active recipients in snapshot date"""
        tx_df["txn_date"] = pd.to_datetime(tx_df["txn_time_utc"]).dt.date
        snapshot_date = pd.to_datetime(snapshot_date).date()
        
        # Canonicalize recipient IDs
        tx_df["recipient_canon_id"] = np.where(
            tx_df["recipient_entity_id"].notna() & (tx_df["recipient_entity_id"] != ""),
            tx_df["recipient_entity_id"],
            np.where(
                tx_df["merchant_id"].notna() & (tx_df["merchant_id"] != ""),
                "MER_" + tx_df["merchant_id"].astype(str),
                "UNK_" + tx_df["to_bank_code"].fillna("") + "_" + tx_df["to_account_number_hash"].fillna("")
            )
        )
        
        active_recipients = tx_df[tx_df["txn_date"] == snapshot_date]["recipient_canon_id"].unique()
        logger.info(f"Found {len(active_recipients)} active recipients on {snapshot_date}")
        
        lookback_start = snapshot_date - pd.Timedelta(days=self.lookback_days)
        
        tx_filtered = tx_df[
            (tx_df["recipient_canon_id"].isin(active_recipients)) &
            (tx_df["txn_date"] >= lookback_start) &
            (tx_df["txn_date"] <= snapshot_date)
        ].copy()
        
        logger.info(f"Processing {len(tx_filtered)} transactions (lookback: {self.lookback_days} days)")
        
        agg = tx_filtered.groupby(["sender_id", "recipient_canon_id"], as_index=False)["amount"].sum()
        
        users = agg["sender_id"].unique()
        items = active_recipients
        
        uid_map = {u: i for i, u in enumerate(users)}
        iid_map = {v: i for i, v in enumerate(items)}
        
        rows = agg["sender_id"].map(uid_map).to_numpy()
        cols = agg["recipient_canon_id"].map(iid_map).to_numpy()
        data = np.log1p(agg["amount"].astype(float)).to_numpy()
        
        A = coo_matrix((data, (rows, cols)), shape=(len(users), len(items))).tocsr()
        
        return A, items
    
    def _compute_svd_embeddings(self, A):
        """Compute SVD embeddings for recipients"""
        k = int(min(self.svd_dim, max(2, min(A.shape) - 1)))
        
        _, s, vt = svds(A, k=k)
        idx = np.argsort(s)[::-1]
        s = s[idx]
        vt = vt[idx, :]
        
        item_emb = vt.T * np.sqrt(s)
        return item_emb
    
    def _extract_metadata_features(self, items, rec_df, mer_df, per_df):
        """Extract and hash metadata features for recipients"""
        ent = rec_df[["recipient_entity_id", "entity_type"]].drop_duplicates()
        mer_min = mer_df[["recipient_entity_id", "mcc"]].drop_duplicates()
        per_min = per_df[["recipient_entity_id", "age_band", "province_code", "occupation_band"]].drop_duplicates()
        
        meta = (ent.merge(mer_min, how="left", on="recipient_entity_id")
                   .merge(per_min, how="left", on="recipient_entity_id")
                   .set_index("recipient_entity_id"))
        
        feat_strings = []
        for rid in items:
            if rid in meta.index:
                r = meta.loc[rid]
                features = []
                features.append(f"etype_{r.get('entity_type', 'UNK')}")
                if pd.notna(r.get("mcc")):
                    features.append(f"mcc_{r['mcc']}")
                if pd.notna(r.get("age_band")):
                    features.append(f"age_{r['age_band']}")
                if pd.notna(r.get("province_code")):
                    features.append(f"prov_{r['province_code']}")
                if pd.notna(r.get("occupation_band")):
                    features.append(f"occ_{r['occupation_band']}")
                feat_strings.append(features if features else ["none"])
            else:
                feat_strings.append(["none"])
        
        hash_features = self.hasher.transform(feat_strings).toarray()
        return hash_features
    
    def generate_features(self, tx_df, rec_df, mer_df, per_df, snapshot_date):
        """Generate combined SVD + metadata features for active recipients"""
        A, items = self._build_interaction_matrix(tx_df, snapshot_date)
        
        svd_emb = self._compute_svd_embeddings(A)
        hash_emb = self._extract_metadata_features(items, rec_df, mer_df, per_df)
        
        E = np.concatenate([svd_emb, hash_emb], axis=1)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        
        return items, E
    
    def _get_dates_to_process(self, tx_df, output_dir, snapshot_date):
        """Determine which dates to process based on existing snapshots"""
        output_dir = Path(output_dir)
        
        if snapshot_date is not None:
            return [snapshot_date]
        
        tx_df["txn_date"] = pd.to_datetime(tx_df["txn_time_utc"]).dt.date
        all_dates = sorted(tx_df["txn_date"].unique())
        all_dates_str = [d.strftime("%Y-%m-%d") for d in all_dates]
        
        existing_dates = set()
        for date_str in all_dates_str:
            snapshot_path = output_dir / f"date={date_str}" / "x_rcv_entity_state.parquet"
            if snapshot_path.exists():
                existing_dates.add(date_str)
        
        dates_to_process = [d for d in all_dates_str if d not in existing_dates]
        
        logger.info(f"Found {len(all_dates_str)} unique transaction dates")
        logger.info(f"Existing snapshots: {len(existing_dates)} dates")
        logger.info(f"To process: {len(dates_to_process)} dates")
        
        if dates_to_process:
            logger.info(f"Processing dates: {dates_to_process[:5]}{'...' if len(dates_to_process) > 5 else ''}")
        
        return dates_to_process
    
    def process_and_save(self, data_dir, output_dir, snapshot_date=None):
        data_dir = Path(data_dir)
        
        tx_df = pd.read_parquet(data_dir / "transactions_landing.parquet")
        rec_df = pd.read_parquet(data_dir / "recipient_entity.parquet")
        mer_df = pd.read_parquet(data_dir / "merchant_profile.parquet")
        per_df = pd.read_parquet(data_dir / "person_profile.parquet")
        
        # Determine dates to process
        dates_to_process = self._get_dates_to_process(tx_df, output_dir, snapshot_date)
        
        if not dates_to_process:
            logger.info("No new dates to process. All snapshots already exist.")
            return None
        
        all_feature_dfs = []
        for date_str in dates_to_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing snapshot date: {date_str}")
            logger.info(f"{'='*60}")
            
            items, embeddings = self.generate_features(tx_df, rec_df, mer_df, per_df, date_str)
            
            feature_df = pd.DataFrame(embeddings, columns=[f"rcv_emb_{i+1}" for i in range(embeddings.shape[1])])
            feature_df.insert(0, "recipient_canon_id", items)
            feature_df["feature_version"] = "v1"
            feature_df["snapshot_date"] = date_str
            feature_df["updated_at"] = datetime.now().isoformat()
            
            # Save daily snapshot
            daily_path = Path(output_dir) / f"date={date_str}" / "x_rcv_entity_state.parquet"
            daily_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_parquet(daily_path, index=False)
            logger.info(f"Generated daily snapshot for {len(items)} recipients -> {daily_path}")
            logger.info(f"SVD dim: {self.svd_dim}, Hash dim: {self.hash_dim}, Total: {embeddings.shape[1]}")
            
            all_feature_dfs.append(feature_df)
        
        # Upsert all processed dates to cumulative feature store
        cumulative_path = Path(output_dir) / "date=ALL" / "x_rcv_entity_state.parquet"
        cumulative_path.parent.mkdir(parents=True, exist_ok=True)
        
        combined_new_df = pd.concat(all_feature_dfs, ignore_index=True)
        
        if cumulative_path.exists():
            # Load existing cumulative data
            existing_df = pd.read_parquet(cumulative_path)
            
            # Remove old records for updated recipients
            all_updated_items = combined_new_df["recipient_canon_id"].unique()
            existing_df = existing_df[~existing_df["recipient_canon_id"].isin(all_updated_items)]
            updated_df = pd.concat([existing_df, combined_new_df], ignore_index=True)
            
            logger.info(f"Updated cumulative store: {len(existing_df)} existing + {len(combined_new_df)} new/updated = {len(updated_df)} total")
        else:
            updated_df = combined_new_df
            logger.info(f"Created new cumulative store with {len(combined_new_df)} recipients")
        
        updated_df.to_parquet(cumulative_path, index=False)
        logger.info(f"Cumulative store saved -> {cumulative_path}")
        
        return combined_new_df


def main(args):
    generator = RecipientFeatureGenerator(
        svd_dim=args.svd_dim, 
        hash_dim=args.hash_dim,
        lookback_days=args.lookback_days
    )
    generator.process_and_save(args.data_dir, args.output_dir, args.snapshot_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate recipient embeddings from interactions and metadata (offline)")
    parser.add_argument("--data-dir", type=str,
                        default=str(ROOT / "data" / "raw"),
                        help="Directory containing input parquet files")
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "data" / "feature_store" / "x_rcv"),
                        help="Output directory for feature store")
    parser.add_argument("--snapshot-date", type=str, default=None,
                        help="Snapshot date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--svd-dim", type=int, default=64,
                        help="SVD embedding dimension")
    parser.add_argument("--hash-dim", type=int, default=32,
                        help="Feature hashing dimension")
    parser.add_argument("--lookback-days", type=int, default=10,
                        help="Number of days to look back for transaction history")
    
    args = parser.parse_args()
    main(args)
