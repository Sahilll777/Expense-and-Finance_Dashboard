# src/data_preprocess.py
import os
import pandas as pd
import re

CANONICAL_COLS = ['transaction_id','date','amount','type','description','mode','category']

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and standardize column names"""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw CSV to canonical schema, clean data, and create features"""
    # Ensure all canonical columns exist
    for c in CANONICAL_COLS:
        if c not in df.columns:
            df[c] = None

    df = df[CANONICAL_COLS]

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Coerce amounts
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['date','amount']).copy()

    # Clean description for NLP
    df['description'] = df['description'].astype(str)
    df['desc_clean'] = df['description'].str.lower()
    df['desc_clean'] = df['desc_clean'].str.replace(r'[^a-z0-9 ]',' ', regex=True)
    df['desc_clean'] = df['desc_clean'].str.replace(r'\s+',' ', regex=True).str.strip()

    # Amount absolute and credit/debit flag
    df['is_credit'] = df['type'].astype(str).str.contains('credit', case=False, na=False)
    df['amount_abs'] = df['amount'].abs()

    # Temporal features
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin(['Saturday','Sunday'])

    return df

def save_processed(df: pd.DataFrame, out_path: str):
    """Save cleaned DataFrame to CSV (auto-creates directories if needed)"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ File saved successfully at: {out_path}")


if __name__ == '__main__':
    import sys
    input_path = sys.argv[1] if len(sys.argv)>1 else 'data/sample_transactions.csv'
    output_path = sys.argv[2] if len(sys.argv)>2 else 'data/processed/transactions_processed.csv'

    df = load_csv(input_path)
    df_clean = preprocess_data(df)
    save_processed(df_clean, output_path)
