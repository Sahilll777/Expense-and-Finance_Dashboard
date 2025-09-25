# tests/test_preprocess.py
import pytest
import pandas as pd
from src.data_preprocess import preprocess_data, load_csv

def test_basic_preprocessing():
    # Load sample CSV
    df = load_csv('data/sample_transactions.csv')
    df_clean = preprocess_data(df)

    # Check if new columns exist
    expected_cols = ['desc_clean','is_credit','amount_abs','month','day_of_week','is_weekend']
    for col in expected_cols:
        assert col in df_clean.columns, f"Missing column: {col}"

    # Check if amounts are positive
    assert df_clean['amount_abs'].min() >= 0, "amount_abs contains negative values"

    # Check if no NaT in dates
    assert df_clean['date'].isnull().sum() == 0, "Some dates are null"

    # Check if desc_clean is lowercase and stripped
    sample_desc = df_clean['desc_clean'].iloc[0]
    assert sample_desc == sample_desc.lower(), "desc_clean not lowercase"
    assert sample_desc == sample_desc.strip(), "desc_clean has leading/trailing spaces"
