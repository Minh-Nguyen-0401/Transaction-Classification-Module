"""
Generate synthetic data for transaction classification pipeline
Creates all necessary tables and files to run the complete pipeline from encoder training to classification.

Usage:
    python generate_synthetic_data.py --num-transactions 100 --num-days 5  # Small test
    python generate_synthetic_data.py --num-transactions 10000 --num-days 30  # Full dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import json
import argparse

# Seed for reproducibility
np.random.seed(42)
random.seed(42)


class SyntheticDataGenerator:
    """Generate synthetic transaction and profile data"""
    
    def __init__(self, num_transactions=100, num_days=5, start_date="2025-11-01"):
        self.num_transactions = num_transactions
        self.num_days = num_days
        self.start_date = pd.to_datetime(start_date)
        
        # Load label map
        label_map_path = Path(__file__).parent / "label_map.json"
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
        # Define enum values from data dictionary
        self.tranx_types = [
            "bill_payment", "cashback", "loan_repayment", "mobile_topup",
            "opensaving", "qrcode_payment", "stock", "transfer_in"
        ]
        
        self.channels = ["MOBILE", "WEB"]
        
        self.categories = list(self.label_map.keys())
        
        self.entity_types = ["MERCHANT", "PERSON"]
        
        self.kyc_tiers = ["L0", "L1", "L2", "L3"]
        self.segments = ["student", "salaried", "self_employed", "merchant", "retired", "other"]
        self.age_bands = ["18-24", "25-34", "35-44", "45-54", "55+", "unk"]
        self.provinces = ["HCM", "HN", "DN", "BD", "HP", "NA"]
        self.income_bands = ["low", "mid", "upper_mid", "high", "unk"]
        self.occupation_bands = ["student", "salaried", "self_employed", "retired", "unk"]
        
        self.bank_codes = ["VCB", "TCB", "ACB", "VTB", "BID", "CTG", "MBB", "TPB"]
        
        # MCC codes by category
        self.mcc_map = {
            "BIL": ["4900", "4814", "4816"],  # Utilities, telecom
            "FOO": ["5812", "5814", "5411"],  # Restaurants, grocery
            "TRN": ["4121", "4131", "5542"],  # Taxi, transport
            "HLT": ["8011", "8021", "8099"],  # Doctors, dentists, health
            "INS": ["6300", "6311", "9311"],  # Insurance
            "SHP": ["5311", "5411", "5651"],  # Department stores, retail
            "ENT": ["7832", "7996", "7999"],  # Movies, entertainment
            "EDU": ["8211", "8220", "8299"],  # Schools, education
            "FIN": ["6010", "6011", "6051"],  # Banks, financial
            "OTH": ["9999", "0000", "7299"]   # Other
        }
        
        # Vietnamese transaction messages by category
        self.msg_templates = {
            "BIL": ["Tien dien thang {}", "Tien nuoc thang {}", "Tien internet", "Thanh toan hoa don"],
            "FOO": ["Mua do an", "Nha hang", "Cafe", "Com trua"],
            "TRN": ["Grab", "Taxi", "Xe om", "Xang xe"],
            "HLT": ["Kham benh", "Thuoc", "Nha khoa", "Bao hiem y te"],
            "INS": ["Bao hiem", "Phi bao hiem", "Dong bao hiem"],
            "SHP": ["Mua sam", "Tien hang", "Shopping"],
            "ENT": ["Ve phim", "Karaoke", "Game", "Giải tri"],
            "EDU": ["Hoc phi", "Sach vo", "Khoa hoc"],
            "FIN": ["Tra gop", "Tiet kiem", "Dau tu"],
            "OTH": ["Chuyen tien", "Gui tien", "Khac"]
        }
        
        # Generate base entities
        self.num_senders = max(10, num_transactions // 20)
        self.num_recipients = max(15, num_transactions // 15)
        self.num_merchants = max(10, self.num_recipients // 2)
        self.num_persons = self.num_recipients - self.num_merchants
        
        print(f"Generating synthetic data:")
        print(f"  Transactions: {num_transactions} over {num_days} days")
        print(f"  Senders: {self.num_senders}")
        print(f"  Recipients: {self.num_recipients} ({self.num_merchants} merchants, {self.num_persons} persons)")
    
    def generate_customer_profile(self):
        """Generate sender/customer profiles"""
        data = {
            "sender_id": [f"CUST_{str(i).zfill(6)}" for i in range(1, self.num_senders + 1)],
            "kyc_tier": np.random.choice(self.kyc_tiers, self.num_senders, p=[0.1, 0.3, 0.4, 0.2]),
            "segment": np.random.choice(self.segments, self.num_senders, p=[0.1, 0.5, 0.2, 0.1, 0.05, 0.05]),
            "age_band": np.random.choice(self.age_bands, self.num_senders, p=[0.15, 0.35, 0.25, 0.15, 0.08, 0.02]),
            "home_province_code": np.random.choice(self.provinces, self.num_senders, p=[0.3, 0.3, 0.15, 0.1, 0.1, 0.05]),
            "income_band": np.random.choice(self.income_bands, self.num_senders, p=[0.25, 0.35, 0.25, 0.1, 0.05])
        }
        return pd.DataFrame(data)
    
    def generate_recipient_entity(self):
        """Generate recipient entities"""
        recipient_ids = []
        entity_types = []
        display_names = []
        
        # Merchants
        merchant_brands = [
            "EVN Ha Noi", "VNPT", "Viettel", "Mobifone", "Grab",
            "Shopeefood", "Highlands Coffee", "Vinmart", "Circle K",
            "Co.opmart", "Bach Hoa Xanh", "CGV Cinemas", "FPT Shop",
            "The Coffee House", "Pizza Hut", "KFC", "Lotte Mart"
        ]
        for i in range(self.num_merchants):
            recipient_ids.append(f"ENT_rcv_{str(i).zfill(5)}")
            entity_types.append("MERCHANT")
            display_names.append(merchant_brands[i % len(merchant_brands)])
        
        # Persons
        first_names = ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu", "Dang", "Bui"]
        middle_names = ["Van", "Thi", "Duc", "Minh", "Thanh", "Anh", "Hong"]
        last_names = ["A", "B", "C", "D", "E", "F", "G", "H", "Linh", "Quan", "Hoa"]
        
        for i in range(self.num_persons):
            recipient_ids.append(f"ENT_rcv_{str(self.num_merchants + i).zfill(5)}")
            entity_types.append("PERSON")
            name = f"{random.choice(first_names)} {random.choice(middle_names)} {random.choice(last_names)}"
            display_names.append(name)
        
        data = {
            "recipient_entity_id": recipient_ids,
            "entity_type": entity_types,
            "primary_display_name": display_names
        }
        return pd.DataFrame(data)
    
    def generate_merchant_profile(self, recipient_df):
        """Generate merchant profiles"""
        merchants = recipient_df[recipient_df["entity_type"] == "MERCHANT"].copy()
        
        data = {
            "merchant_id": [f"MER_{str(i).zfill(5)}" for i in range(len(merchants))],
            "recipient_entity_id": merchants["recipient_entity_id"].values,
            "mcc": [random.choice(self.mcc_map[random.choice(self.categories)]) for _ in range(len(merchants))],
            "brand_name": merchants["primary_display_name"].values
        }
        return pd.DataFrame(data)
    
    def generate_person_profile(self, recipient_df):
        """Generate person profiles"""
        persons = recipient_df[recipient_df["entity_type"] == "PERSON"].copy()
        
        data = {
            "person_id": [f"PERS_{str(i).zfill(5)}" for i in range(len(persons))],
            "recipient_entity_id": persons["recipient_entity_id"].values,
            "age_band": np.random.choice(self.age_bands, len(persons), p=[0.15, 0.35, 0.25, 0.15, 0.08, 0.02]),
            "province_code": np.random.choice(self.provinces, len(persons), p=[0.3, 0.3, 0.15, 0.1, 0.1, 0.05]),
            "occupation_band": np.random.choice(self.occupation_bands, len(persons), p=[0.1, 0.5, 0.25, 0.1, 0.05])
        }
        return pd.DataFrame(data)
    
    def generate_recipient_alias(self, recipient_df):
        """Generate recipient aliases"""
        aliases = []
        
        for _, row in recipient_df.iterrows():
            # Each recipient has 1-3 aliases
            num_aliases = random.randint(1, 3)
            for _ in range(num_aliases):
                alias_type = random.choice(["account_name", "account_number", "phone", "handle"])
                alias_text = row["primary_display_name"]
                
                # Add variations
                if alias_type == "account_number":
                    alias_text = f"{random.randint(100000000, 999999999)}"
                elif alias_type == "phone":
                    alias_text = f"0{random.randint(900000000, 999999999)}"
                elif alias_type == "handle":
                    alias_text = f"@{alias_text.lower().replace(' ', '_')}"
                
                aliases.append({
                    "recipient_entity_id": row["recipient_entity_id"],
                    "alias_text": alias_text,
                    "alias_type": alias_type
                })
        
        return pd.DataFrame(aliases)
    
    def generate_transactions_landing(self, customer_df, recipient_df, merchant_df):
        """Generate main transaction data"""
        transactions = []
        
        # Generate transactions across multiple days
        dates = pd.date_range(self.start_date, periods=self.num_days, freq='D')
        
        for txn_idx in range(self.num_transactions):
            # Select random date
            txn_date = random.choice(dates)
            txn_time = txn_date + timedelta(
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            
            # Select sender
            sender_id = random.choice(customer_df["sender_id"].values)
            
            # Select category (label)
            category = random.choice(self.categories)
            
            # Select tranx_type based on category
            if category == "BIL":
                tranx_type = random.choice(["bill_payment", "mobile_topup"])
            elif category == "FIN":
                tranx_type = random.choice(["loan_repayment", "opensaving", "stock"])
            elif category in ["FOO", "SHP", "ENT"]:
                tranx_type = random.choice(["qrcode_payment", "transfer_in"])
            else:
                tranx_type = random.choice(self.tranx_types)
            
            # Select recipient (bias towards merchants for certain categories)
            recipient_entity_id = None
            merchant_id = None
            
            if category in ["BIL", "FOO", "SHP", "ENT"] and random.random() < 0.8:
                # Merchant transaction - use merchant_id, no recipient_entity_id
                recipient = recipient_df[recipient_df["entity_type"] == "MERCHANT"].sample(1).iloc[0]
                merchant_id = merchant_df[merchant_df["recipient_entity_id"] == recipient["recipient_entity_id"]]["merchant_id"].values[0]
                # recipient_entity_id stays None for merchant transactions
            else:
                # Person transaction or entity-based transaction - use recipient_entity_id, no merchant_id
                recipient = recipient_df.sample(1).iloc[0]
                if recipient["entity_type"] == "PERSON":
                    # Person: use recipient_entity_id
                    recipient_entity_id = recipient["recipient_entity_id"]
                else:
                    # Merchant but using entity ID instead of merchant_id
                    recipient_entity_id = recipient["recipient_entity_id"]
                # merchant_id stays None for entity-based transactions
            
            # Generate amount (realistic distribution)
            if category == "BIL":
                amount = np.random.lognormal(12, 1)  # ~100k-500k VND
            elif category == "FOO":
                amount = np.random.lognormal(11, 0.8)  # ~30k-200k VND
            elif category in ["FIN", "INS"]:
                amount = np.random.lognormal(14, 1.5)  # ~500k-5M VND
            else:
                amount = np.random.lognormal(12.5, 1.2)  # ~200k-1M VND
            
            amount = round(amount / 1000) * 1000  # Round to nearest 1000
            
            # Generate message
            msg_template = random.choice(self.msg_templates[category])
            if "{}" in msg_template:
                msg_content = msg_template.format(random.randint(1, 12))
            else:
                msg_content = msg_template
            
            # Generate transaction
            txn = {
                "txn_id": f"TXN_{str(txn_idx + 1).zfill(8)}",
                "sender_id": sender_id,
                "txn_time_utc": txn_time.isoformat() + "Z",
                "amount": amount,
                "currency": "VND",
                "tranx_type": tranx_type,
                "channel": random.choice(self.channels),
                "category": category,
                "msg_content": msg_content,
                "recipient_entity_id": recipient_entity_id,  # Either this is set...
                "recipient_alias_raw": recipient["primary_display_name"],
                "to_account_number_hash": f"h:acc_{random.randbytes(8).hex()}",
                "to_bank_code": random.choice(self.bank_codes),
                "merchant_id": merchant_id,  # ...or this is set, never both
                "geo_cell_h3_8": f"882a100d{random.randint(0, 9)}bfffff" if random.random() < 0.7 else None
            }
            
            transactions.append(txn)
        
        df = pd.DataFrame(transactions)
        
        # Sort by time
        df = df.sort_values("txn_time_utc").reset_index(drop=True)
        
        return df
    
    def save_all(self, output_dir):
        """Generate and save all datasets"""
        output_dir = Path(output_dir)
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("Generating datasets...")
        print("="*60)
        
        # Generate profiles
        print("\n1. Generating customer profiles...")
        customer_df = self.generate_customer_profile()
        customer_df.to_parquet(raw_dir / "customer_profile.parquet", index=False)
        print(f"   [OK] Saved {len(customer_df)} customer profiles")
        
        print("\n2. Generating recipient entities...")
        recipient_df = self.generate_recipient_entity()
        recipient_df.to_parquet(raw_dir / "recipient_entity.parquet", index=False)
        print(f"   [OK] Saved {len(recipient_df)} recipient entities")
        
        print("\n3. Generating merchant profiles...")
        merchant_df = self.generate_merchant_profile(recipient_df)
        merchant_df.to_parquet(raw_dir / "merchant_profile.parquet", index=False)
        print(f"   [OK] Saved {len(merchant_df)} merchant profiles")
        
        print("\n4. Generating person profiles...")
        person_df = self.generate_person_profile(recipient_df)
        person_df.to_parquet(raw_dir / "person_profile.parquet", index=False)
        print(f"   [OK] Saved {len(person_df)} person profiles")
        
        print("\n5. Generating recipient aliases...")
        alias_df = self.generate_recipient_alias(recipient_df)
        alias_df.to_parquet(raw_dir / "recipient_alias.parquet", index=False)
        print(f"   [OK] Saved {len(alias_df)} recipient aliases")
        
        print("\n6. Generating transactions...")
        transactions_df = self.generate_transactions_landing(customer_df, recipient_df, merchant_df)
        transactions_df.to_parquet(raw_dir / "transactions_landing.parquet", index=False)
        print(f"   [OK] Saved {len(transactions_df)} transactions")
        
        # Print transaction date range
        txn_dates = pd.to_datetime(transactions_df["txn_time_utc"])
        print(f"   Date range: {txn_dates.min().date()} to {txn_dates.max().date()}")
        
        # Print category distribution
        print("\n   Category distribution:")
        for cat, count in transactions_df["category"].value_counts().sort_index().items():
            cat_name = self.label_map[cat]["name"]
            print(f"      {cat} ({cat_name}): {count} ({count/len(transactions_df)*100:.1f}%)")
        
        print("\n" + "="*60)
        print(f"[OK] All datasets saved to: {raw_dir}")
        print("="*60)
        
        # Print summary
        print("\nDataset Summary:")
        print(f"  Customers: {len(customer_df)}")
        print(f"  Recipients: {len(recipient_df)} ({len(merchant_df)} merchants, {len(person_df)} persons)")
        print(f"  Transactions: {len(transactions_df)} over {self.num_days} days")
        print(f"  Categories: {len(self.categories)}")
        
        return {
            "customer_profile": customer_df,
            "recipient_entity": recipient_df,
            "merchant_profile": merchant_df,
            "person_profile": person_df,
            "recipient_alias": alias_df,
            "transactions_landing": transactions_df
        }


def main(args):
    generator = SyntheticDataGenerator(
        num_transactions=args.num_transactions,
        num_days=args.num_days,
        start_date=args.start_date
    )
    
    output_dir = Path(__file__).parent
    datasets = generator.save_all(output_dir)
    
    print("\n" + "="*60)
    print("Ready to run pipeline!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Train context encoder:")
    print("     python src/_build_enc/train_ctx_enc.py")
    print("\n  2. Train sender encoder:")
    print("     python src/_build_enc/train_snd_enc.py")
    print("\n  3. Generate context features WITH TRAINING DATA (--mode train):")
    print("     python src/preprocess/_x_ctx_online.py --mode train")
    print("\n  4. Generate sender features (processes all dates automatically):")
    print("     python src/preprocess/_x_snd_offline.py")
    print("\n  5. Generate recipient features (processes all dates automatically):")
    print("     python src/preprocess/_x_rcv_offline.py")
    print("\n  6. Train classifier (uses ctx_emb.parquet with labels):")
    print("     python src/train_clf/train_main_clf.py")
    print("\n  Note: For inference, run step 3 with --mode inference (default)")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument("--num-transactions", type=int, default=100,
                        help="Number of transactions to generate (default: 100 for testing)")
    parser.add_argument("--num-days", type=int, default=5,
                        help="Number of days to spread transactions across (default: 5)")
    parser.add_argument("--start-date", type=str, default="2025-11-01",
                        help="Start date for transactions (YYYY-MM-DD, default: 2025-11-01)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SYNTHETIC DATA GENERATOR")
    print("="*60)
    print(f"Configuration:")
    print(f"  Transactions: {args.num_transactions}")
    print(f"  Days: {args.num_days}")
    print(f"  Start date: {args.start_date}")
    print("="*60)
    
    main(args)
